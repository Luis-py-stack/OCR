import os
import cv2
import json
import logging
import smtplib
import ssl
import tempfile
import asyncio
from email.message import EmailMessage
from typing import Union, List, Optional

import streamlit as st
import numpy as np
import pymupdf
from google import genai
from google.genai import types

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

PROMPT_PARSER_USUARIO = """
Eres un Asistente Técnico de Control de Obra para Torres y Recipientes de Proceso.
Convierte el reporte de avance del usuario en una lista ordenada y sin duplicados de números enteros de platos completados (#1, #2, ... #N).

REGLAS:
1. Si el usuario da un rango explícito (ej: "del 10 al 15", "10 a 14"), expande todos los números enteros intermedios: [10, 11, 12, 13, 14, 15].
2. Si enumera platos individuales (ej: "completados el 3, 5, 8 y 12"), extrae [3, 5, 8, 12].
3. Si el usuario indica una sección completa o todos, genera la lista completa de enteros correspondiente.
"""

SCHEMA_PARSER_USUARIO = {
    "type": "object",
    "properties": {
        "platos_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Lista de enteros con los IDs de platos terminados."
        }
    },
    "required": ["platos_ids"]
}

PROMPT_ANALISIS_UNIVERSAL = """
Eres un Ingeniero Especialista en Metrología y Planos de Recipientes a Presión (ASME / Koch-Glitsch).

Analiza la vista en elevación horizontal del cilindro ("VESSEL ELEVATION").

OBJETIVOS:
1. Detectar las líneas horizontales continuas interna superior e inferior de la envolvente cilíndrica (y_shell_top, y_shell_bottom) en coordenadas normalizadas (rango 0 a 1000).
2. Identificar TODOS los bloques o tramos de platos presentes en el plano (ej. un único tramo de 1-20, o múltiples tramos 1-20, 21-37, etc.).
3. Para cada bloque o tramo detectado:
   - plato_inicio: Número entero del primer plato del tramo (ej. 1).
   - plato_fin: Número entero del último plato del tramo (ej. 20 o 90).
   - x_primer_plato: Coordenada X normalizada (0 a 1000) exactamente sobre la LÍNEA VERTICAL del plato inicial.
   - x_ultimo_plato: Coordenada X normalizada (0 a 1000) exactamente sobre la LÍNEA VERTICAL del plato final.

REGLA CRÍTICA DE ALINEACIÓN (MUY IMPORTANTE):
- En estos diagramas técnicos, cada plato tiene un rótulo con su número (ej. '#1', '#2', ..., '#10', ..., '#20').
- La línea física vertical del plato (la estructura sólida que cruza el cilindro y que debe ser tachada con la línea roja) está situada **INMEDIATAMENTE A LA DERECHA** del texto del número.
- NUNCA sitúes x_primer_plato a la izquierda del texto '#1' ni sobre el texto. Debe coincidir con la línea vertical dibujada a su DERECHA.
- Igualmente, x_ultimo_plato DEBE ser la línea vertical situada inmediatamente a la DERECHA del rótulo del último plato (ej. a la derecha de '#20').
"""

SCHEMA_ANALISIS_UNIVERSAL = {
    "type": "object",
    "properties": {
        "y_shell_top": {"type": "number"},
        "y_shell_bottom": {"type": "number"},
        "bloques_platos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "plato_inicio": {"type": "integer"},
                    "plato_fin": {"type": "integer"},
                    "x_primer_plato": {"type": "number"},
                    "x_ultimo_plato": {"type": "number"}
                },
                "required": ["plato_inicio", "plato_fin", "x_primer_plato", "x_ultimo_plato"]
            }
        }
    },
    "required": ["y_shell_top", "y_shell_bottom", "bloques_platos"]
}

def ajustar_a_linea_cad(img_gray, x_teorico, y_top, y_bot, radio_busqueda=20):
    alto, ancho = img_gray.shape
    x_min = max(0, x_teorico - 5)
    x_max = min(ancho - 1, x_teorico + radio_busqueda)

    h = y_bot - y_top
    y_ini = y_top + int(h * 0.15)
    y_fin = y_bot - int(h * 0.15)

    if y_fin <= y_ini or x_max <= x_min:
        return x_teorico

    roi = img_gray[y_ini:y_fin, x_min:x_max + 1]
    mascara_lineas = np.where(roi < 140, 255 - roi, 0)
    perfil_vertical = mascara_lineas.sum(axis=0)

    if perfil_vertical.max() > 0:
        offset_optimo = int(perfil_vertical.argmax())
        return x_min + offset_optimo

    return x_teorico


def trazar_avance_opencv(ruta_in, ruta_out, metadata_geometria, platos_a_marcar):
    img = cv2.imread(ruta_in)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_in}")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    alto, ancho = img.shape[:2]

    def to_px_y(val):
        return int(round((val / 1000.0) * alto))
    def to_px_x(val):
        return int(round((val / 1000.0) * ancho))

    y_top = to_px_y(metadata_geometria["y_shell_top"])
    y_bot = to_px_y(metadata_geometria["y_shell_bottom"])
    color_rojo = (0, 0, 255)
    grosor_linea = 3
    total_trazados = 0

    bloques = metadata_geometria.get("bloques_platos", [])
    platos_set = set(platos_a_marcar)

    for bloque in bloques:
        p_ini = bloque["plato_inicio"]
        p_fin = bloque["plato_fin"]
        total_espacios = p_fin - p_ini

        if total_espacios <= 0:
            continue

        x_ini_px = to_px_x(bloque["x_primer_plato"])
        x_fin_px = to_px_x(bloque["x_ultimo_plato"])
        step_px = (x_fin_px - x_ini_px) / float(total_espacios)

        for p_num in range(p_ini, p_fin + 1):
            if p_num in platos_set:
                offset_idx = p_num - p_ini
                x_calc = int(round(x_ini_px + offset_idx * step_px))
                x_pos = ajustar_a_linea_cad(img_gray, x_calc, y_top, y_bot, radio_busqueda=22)

                cv2.line(img, (x_pos, y_top), (x_pos, y_bot), color_rojo, thickness=grosor_linea, lineType=cv2.LINE_AA)
                total_trazados += 1

    cv2.imwrite(ruta_out, img)
    return total_trazados

def enviar_correo(
    email_sender: str,
    email_password: str,
    smtp_server: str,
    smtp_port: int,
    destinatarios: Union[str, List[str]],
    asunto: str,
    cuerpo: str,
    ruta_adjunto: Optional[str] = None
) -> bool:
    if isinstance(destinatarios, str):
        destinatarios = [destinatarios]

    em = EmailMessage()
    em["From"] = email_sender
    em["To"] = ", ".join(destinatarios)
    em["Subject"] = asunto
    em.set_content(cuerpo)

    if ruta_adjunto and os.path.exists(ruta_adjunto):
        with open(ruta_adjunto, "rb") as f:
            contenido_archivo = f.read()
            nombre_archivo = os.path.basename(ruta_adjunto)

        em.add_attachment(
            contenido_archivo,
            maintype="application",
            subtype="pdf",
            filename=nombre_archivo
        )

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls(context=context)
            smtp.ehlo()
            smtp.login(email_sender, email_password)
            smtp.send_message(em, to_addrs=destinatarios)
        return True
    except Exception as e:
        st.error(f"Error al enviar correo vía SMTP: {e}")
        return False

async def procesar_plano_universal(
    pdf_in_bytes: bytes,
    nombre_archivo_original: str,
    reporte_usuario: str,
    client_agente_a: genai.Client,
    client_agente_b: genai.Client,
    modelo_gemini: str,
    temp_dir: str
):
    resp_user = await client_agente_a.aio.models.generate_content(
        model=modelo_gemini,
        contents=[
            types.Part.from_text(text=f"Reporte del usuario: {reporte_usuario}"),
            types.Part.from_text(text=PROMPT_PARSER_USUARIO)
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SCHEMA_PARSER_USUARIO
        )
    )
    datos_avance = json.loads(resp_user.text)
    platos_solicitados = sorted(datos_avance.get("platos_ids", []))

    doc = pymupdf.open(stream=pdf_in_bytes, filetype="pdf")
    pagina = doc[0]
    pix = pagina.get_pixmap(matrix=pymupdf.Matrix(3.0, 3.0))
    img_temp = os.path.join(temp_dir, "temp_plano_original.png")
    pix.save(img_temp)

    with open(img_temp, "rb") as f:
        img_bytes = f.read()

    resp_geom = await client_agente_b.aio.models.generate_content(
        model=modelo_gemini,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            types.Part.from_text(text=PROMPT_ANALISIS_UNIVERSAL)
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SCHEMA_ANALISIS_UNIVERSAL
        )
    )
    geometria_plano = json.loads(resp_geom.text)

    img_trazada = os.path.join(temp_dir, "temp_plano_trazado.png")
    total_trazados = trazar_avance_opencv(img_temp, img_trazada, geometria_plano, platos_solicitados)

    pdf_out = os.path.join(temp_dir, f"Avance_{nombre_archivo_original}")
    doc_nuevo = pymupdf.open()
    pix_mod = pymupdf.Pixmap(img_trazada)
    nueva_pag = doc_nuevo.new_page(width=pix_mod.width, height=pix_mod.height)
    nueva_pag.insert_image(nueva_pag.rect, filename=img_trazada)
    doc_nuevo.save(pdf_out)
    doc_nuevo.close()
    doc.close()

    with open(pdf_out, "rb") as f:
        pdf_out_bytes = f.read()

    return platos_solicitados, geometria_plano, total_trazados, img_trazada, pdf_out, pdf_out_bytes

def main():
    st.set_page_config(page_title="Trazado de Avance de Platos", layout="wide")
    st.title("Sistema Universal de Trazado de Avance de Platos")
    st.caption("Calibración con Snap-to-CAD asistida por IA multimodal")

    with st.sidebar:
        st.header("🔑 Configuración de APIs")
        api_key_a = st.text_input("Gemini API Key (Agente A)", type="password")
        api_key_b = st.text_input("Gemini API Key (Agente B - opcional)", type="password", help="Si se deja vacía, se usará la API Key del Agente A.")
        modelo_gemini = st.text_input("Modelo Gemini", value="gemini-3.6-flash")

        st.header("✉️ Configuración de Correo SMTP")
        email_sender = st.text_input("Correo remitente", value="agent.holtmontai@gmail.com")
        email_password = st.text_input("Contraseña de aplicación", type="password")
        smtp_server = st.text_input("Servidor SMTP", value="smtp.gmail.com")
        smtp_port = st.number_input("Puerto SMTP", value=587)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Carga de Documento")
        archivo_pdf = st.file_uploader("Selecciona el plano técnico en PDF", type=["pdf"])

        st.subheader("2. Reporte de Obra")
        mensaje_usuario = st.text_area(
            "Descripción del avance de platos",
            value="Ya están listos los platos #10, #11, #12, #13, #14 y #15",
            height=120
        )

        st.subheader("3. Notificación (Opcional)")
        correo_destino = st.text_input("Destinatario del reporte por correo")

        procesar_btn = st.button("Procesar y Trazar Avance", type="primary", use_container_width=True)

    with col2:
        st.subheader("Vista Previa y Resultados")
        resultado_placeholder = st.empty()

    if procesar_btn:
        if not api_key_a:
            st.error("Es obligatorio ingresar al menos la API Key del Agente A en la barra lateral.")
            return

        if not archivo_pdf:
            st.error("Debes cargar un archivo PDF de plano técnico.")
            return

        key_b = api_key_b.strip() if api_key_b.strip() else api_key_a.strip()
        client_a = genai.Client(api_key=api_key_a.strip())
        client_b = genai.Client(api_key=key_b)

        with st.spinner("Procesando documento técnico y calibrando trazado..."):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_bytes = archivo_pdf.read()

                    platos, geom, total_trazados, img_trazada_path, pdf_out_path, pdf_out_bytes = asyncio.run(
                        procesar_plano_universal(
                            pdf_in_bytes=pdf_bytes,
                            nombre_archivo_original=archivo_pdf.name,
                            reporte_usuario=mensaje_usuario,
                            client_agente_a=client_a,
                            client_agente_b=client_b,
                            modelo_gemini=modelo_gemini,
                            temp_dir=temp_dir
                        )
                    )

                    with col2:
                        st.image(img_trazada_path, caption=f"Trazado generado ({total_trazados} platos alineados)", use_container_width=True)

                        st.download_button(
                            label="📥 Descargar Plano Trazado (PDF)",
                            data=pdf_out_bytes,
                            file_name=f"Trazado_{archivo_pdf.name}",
                            mime="application/pdf",
                            use_container_width=True
                        )

                    st.success(f"Proceso concluido exitosamente: {total_trazados} platos marcados ({platos}).")

                    with st.expander("Ver metadata geométrica detectada"):
                        st.json(geom)

                    # Envío de correo si fue solicitado
                    if correo_destino.strip():
                        if not email_password:
                            st.warning("No se proporcionó la contraseña de aplicación SMTP en la barra lateral. No se envió el correo.")
                        else:
                            platos_str = ", ".join(f"#{p}" for p in platos) if platos else "Ninguno"
                            asunto_correo = f"Reporte de Avance - Trazado de Platos [{archivo_pdf.name}]"
                            cuerpo_correo = f"""Estimado equipo de Control de Obra / Supervisión:

Se ha completado el procesamiento y marcado automático de avance físico sobre el plano técnico.

RESUMEN DETALLADO DE OPERACIÓN:
----------------------------------------------------------------------
- Reporte recibido: "{mensaje_usuario}"
- Platos interpretados por IA ({len(platos)} en total): {platos_str}
- Líneas estructurales trazadas en CAD (OpenCV): {total_trazados}
- Archivo fuente: {archivo_pdf.name}
----------------------------------------------------------------------

El plano actualizado con los platos completados se encuentra adjunto para su revisión.

Atentamente,
Sistema Automatizado HoltmontAI
"""
                            enviado = enviar_correo(
                                email_sender=email_sender,
                                email_password=email_password,
                                smtp_server=smtp_server,
                                smtp_port=int(smtp_port),
                                destinatarios=correo_destino.strip(),
                                asunto=asunto_correo,
                                cuerpo=cuerpo_correo,
                                ruta_adjunto=pdf_out_path
                            )
                            if enviado:
                                st.info(f"Notificación enviada por correo a: {correo_destino.strip()}")

            except Exception as e:
                st.error(f"Ocurrió un error durante el procesamiento: {e}")

if __name__ == "__main__":
    main()

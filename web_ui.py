"""
web_ui.py - Interfaz web optimizada para hablar con Aitana usando Gradio.
Compatible con Gradio 6.
"""

import argparse
import gradio as gr
from aitana_core import AitanaCore, DEFAULT_MODEL

# instancia global
aitana = None


def respond(message: str, chat_history: list):

    global aitana

    if not message:
        return "", chat_history

    if chat_history is None:
        chat_history = []

    message = message.strip()

    # ---------------------------
    # COMANDO: /estado
    # ---------------------------

    if message.lower() == "/estado":

        status = aitana.get_status()

        bot_message = "### Estado de Aitana\n"

        for key, value in status.items():
            bot_message += f"- **{key}**: {value}\n"

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})

        return "", chat_history

    # ---------------------------
    # COMANDO: /limpiar
    # ---------------------------

    if message.lower() == "/limpiar":

        aitana.clear_history()

        return "", []

    # ---------------------------
    # COMANDO: /memoria
    # ---------------------------

    if message.lower().startswith("/memoria "):

        text = message[9:].strip()

        success = aitana.add_memory(text)

        result = (
            "Memoria guardada correctamente."
            if success
            else "No se pudo guardar la memoria."
        )

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": result})

        return "", chat_history

    # ---------------------------
    # CHAT NORMAL
    # ---------------------------

    chat_history.append({"role": "user", "content": message})

    # indicador de escritura
    chat_history.append({"role": "assistant", "content": "Aitana está escribiendo..."})

    yield "", chat_history

    try:

        response = aitana.chat(message)

    except Exception as e:

        response = f"Error generando respuesta:\n{str(e)}"

    # reemplazar indicador
    chat_history[-1] = {"role": "assistant", "content": response}

    yield "", chat_history


def upload_file(file):

    global aitana

    if file is None:
        return "No se seleccionó ningún archivo."

    try:

        success = aitana.add_memory_file(file.name)

        if success:
            return f"Archivo procesado correctamente:\n{file.name}"

        return "No se pudo procesar el archivo."

    except Exception as e:

        return f"Error procesando archivo:\n{str(e)}"


def create_ui():

    with gr.Blocks(title="Aitana - Tu Amiga Virtual") as app:

        gr.Markdown(
            """
# ✨ Aitana - Tu Amiga Virtual

Escribe un mensaje para hablar con Aitana.

**Comandos disponibles**

- `/estado` → muestra información del sistema
- `/limpiar` → limpia la conversación
- `/memoria texto` → guarda algo en memoria
"""
        )

        chatbot = gr.Chatbot(
    label="Conversación",
    height=500,
    avatar_images=(
        None,
        "https://api.dicebear.com/7.x/avataaars/svg?seed=Aitana&backgroundColor=ffd5dc",
    ),
)

        with gr.Row():

            msg = gr.Textbox(
                placeholder="Escribe tu mensaje...",
                scale=4,
                container=False,
            )

            send_btn = gr.Button(
                "Enviar",
                variant="primary",
                scale=1,
            )

        with gr.Accordion("Agregar archivo a la memoria", open=False):

            file_upload = gr.File(
                label="Sube un archivo (.txt, .md, .csv)",
                file_types=[".txt", ".md", ".csv"],
            )

            upload_btn = gr.Button("Procesar archivo")

            upload_status = gr.Textbox(
                label="Estado",
                interactive=False,
            )

        # eventos

        msg.submit(
            respond,
            [msg, chatbot],
            [msg, chatbot],
        )

        send_btn.click(
            respond,
            [msg, chatbot],
            [msg, chatbot],
        )

        upload_btn.click(
            upload_file,
            [file_upload],
            [upload_status],
        )

    return app


def main():

    parser = argparse.ArgumentParser(description="Interfaz web para Aitana.")

    parser.add_argument(
        "--modelo",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Modelo de Ollama a usar (default: {DEFAULT_MODEL})",
    )

    parser.add_argument(
        "--sin-rag",
        action="store_true",
        help="Desactivar memoria a largo plazo",
    )

    parser.add_argument(
        "--puerto",
        "-p",
        type=int,
        default=7860,
        help="Puerto del servidor",
    )

    args = parser.parse_args()

    global aitana

    print("Iniciando a Aitana...")

    aitana = AitanaCore(
        model=args.modelo,
        use_rag=not args.sin_rag,
    )

    app = create_ui()

    print(f"\nInterfaz disponible en:")
    print(f"http://localhost:{args.puerto}")

    app.launch(
        server_port=args.puerto,
        theme=gr.themes.Soft(
            primary_hue="pink",
            secondary_hue="purple",
        ),
        css="""
        .gradio-container {
            max-width: 900px !important;
            margin: auto;
        }

        footer {
            display: none !important;
        }
        """,
        share=False,
    )


if __name__ == "__main__":
    main()
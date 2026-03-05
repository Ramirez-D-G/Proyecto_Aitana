"""
web_ui.py - Interfaz web para hablar con Aitana usando Gradio.

Ejecuta este archivo para abrir una interfaz web en tu navegador.
"""

import argparse
import gradio as gr
from aitana_core import AitanaCore, DEFAULT_MODEL


# Variable global para la instancia de Aitana
aitana = None


def respond(message: str, chat_history: list) -> tuple:
    """
    Procesa un mensaje del usuario y devuelve la respuesta de Aitana.

    Args:
        message: Mensaje del usuario.
        chat_history: Historial de mensajes en formato Gradio.

    Returns:
        Tuple con (campo de texto vacío, historial actualizado).
    """
    global aitana

    # Comandos especiales
    if message.strip().lower() == "/estado":
        status = aitana.get_status()
        bot_message = "--- Estado de Aitana ---\n"
        for key, value in status.items():
            bot_message += f"  {key}: {value}\n"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    if message.strip().lower() == "/limpiar":
        aitana.clear_history()
        return "", []

    if message.strip().lower().startswith("/memoria "):
        text = message.strip()[9:]
        success = aitana.add_memory(text)
        result = "Memoria guardada correctamente." if success else "No se pudo guardar."
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": result})
        return "", chat_history

    # Mensaje normal: obtener respuesta de Aitana
    response = aitana.chat(message)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})

    return "", chat_history


def upload_file(file) -> str:
    """
    Procesa un archivo subido y lo agrega a la memoria de Aitana.

    Args:
        file: Archivo subido por el usuario.

    Returns:
        Mensaje de estado.
    """
    global aitana
    if file is None:
        return "No se seleccionó ningún archivo."

    success = aitana.add_memory_file(file.name)
    if success:
        return f"Archivo '{file.name}' procesado y guardado en la memoria de Aitana."
    return f"No se pudo procesar el archivo '{file.name}'."


def create_ui() -> gr.Blocks:
    """Crea la interfaz web con Gradio."""

    with gr.Blocks(
        title="Aitana - Tu Amiga Virtual",
        theme=gr.themes.Soft(
            primary_hue="pink",
            secondary_hue="purple",
        ),
        css="""
        .gradio-container { max-width: 800px !important; margin: auto; }
        footer { display: none !important; }
        """
    ) as app:
        gr.Markdown(
            """
            # ✨ Aitana - Tu Amiga Virtual ✨
            *Escribe un mensaje para hablar con Aitana. Comandos: /estado, /limpiar, /memoria <texto>*
            """
        )

        chatbot = gr.Chatbot(
            label="Conversación",
            height=500,
            type="messages",
            avatar_images=(None, "https://api.dicebear.com/7.x/avataaars/svg?seed=Aitana&backgroundColor=ffd5dc"),
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Tu mensaje",
                placeholder="Escribe aquí...",
                scale=4,
                container=False,
            )
            send_btn = gr.Button("Enviar", variant="primary", scale=1)

        with gr.Accordion("Agregar archivo a la memoria", open=False):
            file_upload = gr.File(
                label="Sube un archivo de texto (.txt) para que Aitana lo recuerde",
                file_types=[".txt", ".md", ".csv"],
            )
            upload_btn = gr.Button("Procesar archivo")
            upload_status = gr.Textbox(label="Estado", interactive=False)

        # Eventos
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        send_btn.click(respond, [msg, chatbot], [msg, chatbot])
        upload_btn.click(upload_file, [file_upload], [upload_status])

    return app


def main():
    """Función principal - inicia la interfaz web."""

    parser = argparse.ArgumentParser(description="Interfaz web para Aitana.")
    parser.add_argument(
        "--modelo", "-m",
        default=DEFAULT_MODEL,
        help=f"Modelo de Ollama a usar (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--sin-rag",
        action="store_true",
        help="Desactivar la memoria a largo plazo (RAG)"
    )
    parser.add_argument(
        "--puerto", "-p",
        type=int,
        default=7860,
        help="Puerto para la interfaz web (default: 7860)"
    )
    args = parser.parse_args()

    # Inicializar Aitana
    global aitana
    print("Iniciando a Aitana...")
    aitana = AitanaCore(model=args.modelo, use_rag=not args.sin_rag)

    # Crear y lanzar la interfaz
    app = create_ui()
    print(f"\nAbriendo interfaz web en http://localhost:{args.puerto}")
    app.launch(server_port=args.puerto, share=False)


if __name__ == "__main__":
    main()

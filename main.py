"""
main.py - Interfaz de texto por consola para hablar con Aitana.

Ejecuta este archivo para iniciar una conversación con Aitana por la terminal.
Escribe tus mensajes y presiona Enter. Escribe 'salir' para terminar.
"""

import sys
import argparse
from aitana_core import AitanaCore, DEFAULT_MODEL


# Colores para la terminal (funciona en la mayoría de terminales modernas)
class Colors:
    AITANA = "\033[95m"    # Magenta/rosa para Aitana
    USER = "\033[96m"      # Cyan para el usuario
    SYSTEM = "\033[93m"    # Amarillo para mensajes del sistema
    RESET = "\033[0m"      # Reset
    BOLD = "\033[1m"       # Negrita


def print_banner():
    """Muestra el banner de bienvenida."""
    banner = f"""
{Colors.AITANA}{Colors.BOLD}
    ╔═══════════════════════════════════════╗
    ║          ✨ AITANA ✨                 ║
    ║       Tu amiga virtual                ║
    ╚═══════════════════════════════════════╝
{Colors.RESET}"""
    print(banner)


def print_help():
    """Muestra los comandos disponibles."""
    print(f"""
{Colors.SYSTEM}--- Comandos disponibles ---
  /salir           - Terminar la conversación
  /estado          - Ver estado de Aitana (modelo, memoria, etc.)
  /limpiar         - Borrar el historial de conversación
  /memoria <texto> - Agregar información a la memoria a largo plazo
  /archivo <ruta>  - Agregar un archivo a la memoria a largo plazo
  /ayuda           - Mostrar estos comandos
{Colors.RESET}""")


def main():
    """Función principal - inicia la conversación con Aitana por consola."""

    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Habla con Aitana, tu amiga virtual.")
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
    args = parser.parse_args()

    # Mostrar banner
    print_banner()
    print(f"{Colors.SYSTEM}Iniciando a Aitana...{Colors.RESET}\n")

    # Inicializar Aitana
    try:
        aitana = AitanaCore(model=args.modelo, use_rag=not args.sin_rag)
    except Exception as e:
        print(f"{Colors.SYSTEM}[Error] No se pudo inicializar Aitana: {e}{Colors.RESET}")
        sys.exit(1)

    print(f"\n{Colors.SYSTEM}Aitana está lista. Escribe tu mensaje y presiona Enter.")
    print(f"Escribe /ayuda para ver los comandos disponibles.{Colors.RESET}\n")

    # Loop principal de conversación
    while True:
        try:
            # Leer input del usuario
            user_input = input(f"{Colors.USER}{Colors.BOLD}Tú: {Colors.RESET}").strip()

            # Ignorar entradas vacías
            if not user_input:
                continue

            # Procesar comandos especiales
            if user_input.lower() in ["/salir", "salir", "exit", "quit"]:
                print(f"\n{Colors.AITANA}Aitana: ¡Hasta luego! Cuídate mucho, ya sabes que aquí estaré "
                      f"cuando me necesites 💖{Colors.RESET}\n")
                break

            if user_input.lower() == "/ayuda":
                print_help()
                continue

            if user_input.lower() == "/estado":
                status = aitana.get_status()
                print(f"\n{Colors.SYSTEM}--- Estado de Aitana ---")
                for key, value in status.items():
                    print(f"  {key}: {value}")
                print(f"{Colors.RESET}")
                continue

            if user_input.lower() == "/limpiar":
                aitana.clear_history()
                print(f"{Colors.SYSTEM}Historial borrado. Conversación reiniciada.{Colors.RESET}\n")
                continue

            if user_input.lower().startswith("/memoria "):
                text = user_input[9:].strip()
                if text:
                    success = aitana.add_memory(text)
                    if success:
                        print(f"{Colors.SYSTEM}Memoria guardada correctamente.{Colors.RESET}\n")
                    else:
                        print(f"{Colors.SYSTEM}No se pudo guardar la memoria.{Colors.RESET}\n")
                continue

            if user_input.lower().startswith("/archivo "):
                filepath = user_input[9:].strip()
                if filepath:
                    success = aitana.add_memory_file(filepath)
                    if success:
                        print(f"{Colors.SYSTEM}Archivo procesado y guardado en memoria.{Colors.RESET}\n")
                    else:
                        print(f"{Colors.SYSTEM}No se pudo procesar el archivo.{Colors.RESET}\n")
                continue

            # Enviar mensaje a Aitana y mostrar respuesta en streaming
            print(f"\n{Colors.AITANA}{Colors.BOLD}Aitana: {Colors.RESET}{Colors.AITANA}", end="", flush=True)

            for token in aitana.chat_stream(user_input):
                print(token, end="", flush=True)

            print(f"{Colors.RESET}\n")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.AITANA}Aitana: ¡Ey! No te vayas así de golpe 😄 "
                  f"¡Cuídate mucho! 💖{Colors.RESET}\n")
            break

        except EOFError:
            print(f"\n{Colors.AITANA}Aitana: ¡Hasta pronto! 💖{Colors.RESET}\n")
            break


if __name__ == "__main__":
    main()

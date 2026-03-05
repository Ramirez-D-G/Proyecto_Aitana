"""
feed_documents.py - Script para alimentar documentos a la memoria de Aitana.

Uso:
    python feed_documents.py archivo1.txt archivo2.txt ...
    python feed_documents.py --carpeta mis_documentos/
    python feed_documents.py --texto "Información que quiero que Aitana recuerde"
    python feed_documents.py --stats   (ver estadísticas de la memoria)
    python feed_documents.py --clear   (borrar toda la memoria)
"""

import os
import sys
import argparse
from rag_engine import RAGEngine


def main():
    parser = argparse.ArgumentParser(
        description="Alimenta documentos a la memoria a largo plazo de Aitana."
    )
    parser.add_argument(
        "archivos",
        nargs="*",
        help="Archivos de texto a agregar a la memoria"
    )
    parser.add_argument(
        "--carpeta", "-c",
        help="Carpeta con archivos de texto para agregar"
    )
    parser.add_argument(
        "--texto", "-t",
        help="Texto directo para agregar a la memoria"
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Mostrar estadísticas de la memoria"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Borrar toda la memoria a largo plazo"
    )

    args = parser.parse_args()

    # Inicializar el motor RAG
    rag = RAGEngine()

    # Mostrar estadísticas
    if args.stats:
        stats = rag.get_stats()
        print("\n--- Estadísticas de la memoria ---")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()
        return

    # Borrar memoria
    if args.clear:
        confirm = input("¿Estás seguro de que quieres borrar TODA la memoria? (si/no): ")
        if confirm.lower() in ["si", "sí", "s", "yes", "y"]:
            rag.clear()
            print("Memoria borrada.")
        else:
            print("Operación cancelada.")
        return

    # Agregar texto directo
    if args.texto:
        count = rag.add_document(args.texto, source="texto_directo")
        print(f"Texto agregado: {count} chunks.")

    # Agregar archivos individuales
    total_chunks = 0
    for filepath in args.archivos:
        count = rag.add_file(filepath)
        total_chunks += count

    # Agregar carpeta completa
    if args.carpeta:
        if not os.path.isdir(args.carpeta):
            print(f"Error: '{args.carpeta}' no es una carpeta válida.")
            sys.exit(1)

        extensions = {'.txt', '.md', '.csv', '.log'}
        for filename in sorted(os.listdir(args.carpeta)):
            _, ext = os.path.splitext(filename)
            if ext.lower() in extensions:
                filepath = os.path.join(args.carpeta, filename)
                count = rag.add_file(filepath)
                total_chunks += count

    # Resumen
    if total_chunks > 0 or args.texto:
        print(f"\n--- Resumen ---")
        print(f"Total de chunks agregados: {total_chunks}")
        stats = rag.get_stats()
        print(f"Total en memoria: {stats['total_chunks']} chunks")
    elif not args.archivos and not args.carpeta and not args.texto:
        parser.print_help()


if __name__ == "__main__":
    main()

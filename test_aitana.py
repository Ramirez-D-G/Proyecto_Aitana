"""
test_aitana.py - Script de prueba para verificar que Aitana funciona.

Ejecuta este script para comprobar que todos los componentes están
correctamente instalados y funcionando.
"""

import sys


def test_imports():
    """Verifica que todas las dependencias están instaladas."""
    print("1. Verificando dependencias...")
    errors = []

    try:
        import requests
        print("   - requests: OK")
    except ImportError:
        errors.append("requests")
        print("   - requests: FALTA")

    try:
        import chromadb
        print("   - chromadb: OK")
    except ImportError:
        errors.append("chromadb")
        print("   - chromadb: FALTA")

    try:
        from sentence_transformers import SentenceTransformer
        print("   - sentence-transformers: OK")
    except ImportError:
        errors.append("sentence-transformers")
        print("   - sentence-transformers: FALTA")

    if errors:
        print(f"\n   FALTAN dependencias: {', '.join(errors)}")
        print("   Ejecuta: pip install -r requirements.txt")
        return False

    print("   Todas las dependencias OK.\n")
    return True


def test_ollama():
    """Verifica que Ollama está corriendo y el modelo disponible."""
    import requests

    print("2. Verificando conexión con Ollama...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()

        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]

        print(f"   Ollama conectado. Modelos disponibles: {', '.join(model_names)}")

        llama_models = [m for m in model_names if "llama" in m.lower()]

        if llama_models:
            print(f"   Modelos Llama encontrados: {', '.join(llama_models)}")
        else:
            print("   AVISO: No se encontró ningún modelo Llama.")
            print("   Ejecuta: ollama pull llama3.1:8b-instruct-q4_0")

        return True

    except requests.ConnectionError:
        print("   ERROR: No se pudo conectar con Ollama.")
        print("   Asegúrate de que Ollama está corriendo (ejecuta 'ollama serve').")
        return False

    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def test_rag():
    """Prueba el motor RAG."""
    print("\n3. Probando motor RAG...")

    try:
        from rag_engine import RAGEngine

        rag = RAGEngine()

        # Documento de prueba con información del usuario
        test_text = (
            "Mi mejor amigo se llama (Tu nombre), aunque la mayoría de sus amigos lo conocen como (Tu apodo). "
            "Nació el DD/MM/AA en (Ligar de nacimiento). "
            "Es originario de (Lugar de donde eres originario), también conocida como (Como se conoce comunmente), en el pueblo de (Tu Pueblo) en (Tu estado). "
            "Actualmente estudia (Lo que estudias) en la (En donde estudias). "
            "Le gusta mucho (Tus gustos). "
            "En sus tiempos libres (Lo que haces en tus pasatiempos libres). "
            "Le encanta (Lo que te encanta). "
            "También disfruta (Lo que disfrutas). "
            "Sus artistas favoritos son (Tus artistas favoritos). "
            "Entre sus videojuegos favoritos están (Tus videojuegos favoritos). "
            "A mi amigo (Tu apodo) también le apasiona (Lo que te apaciona). "
            "Tiene (Tu(s) mascota(s)). "
            "Uno de los sueños de (Tu apodo) es (Tu sueño)."
        )

        count = rag.add_document(test_text, source="test")

        print(f"   Documento de prueba agregado: {count} chunks")

        # Búsqueda en memoria
        results = rag.search("¿Cómo se llama el gato de Chalo?")

        if results:
            print("   Búsqueda exitosa. Resultado encontrado:")
            print(f"   {results[0][:120]}...")
        else:
            print("   AVISO: La búsqueda no devolvió resultados.")

        # Estadísticas
        stats = rag.get_stats()

        print(f"   Total chunks en memoria: {stats['total_chunks']}")

        print("   Motor RAG: OK\n")

        return True

    except Exception as e:
        print(f"   ERROR en RAG: {e}")
        return False


def test_chat():
    """Prueba una conversación básica con Aitana."""
    print("4. Probando conversación con Aitana...")

    try:
        from aitana_core import AitanaCore

        aitana = AitanaCore(use_rag=True)

        print("   Enviando mensaje de prueba...")

        response = aitana.chat("Hola, ¿cómo te llamas?")

        print(f"   Respuesta de Aitana: {response[:150]}...")

        if response.startswith("[Error]"):
            print("   AVISO: Hubo un error en la respuesta.")
            return False

        print("   Conversación: OK\n")

        return True

    except Exception as e:
        print(f"   ERROR en chat: {e}")
        return False


def main():

    print("=" * 50)
    print("  PRUEBA DE AITANA - Tu Amiga Virtual")
    print("=" * 50)
    print()

    # Test 1: Imports
    if not test_imports():
        print("\nInstala las dependencias primero.")
        sys.exit(1)

    # Test 2: Ollama
    ollama_ok = test_ollama()

    # Test 3: RAG
    rag_ok = test_rag()

    # Test 4: Chat (solo si Ollama está disponible)
    chat_ok = False

    if ollama_ok:
        chat_ok = test_chat()
    else:
        print("\n4. Saltando prueba de chat (Ollama no disponible).\n")

    # Resumen
    print("=" * 50)
    print("  RESUMEN")
    print("=" * 50)

    print(f"  Dependencias:  {'OK' if True else 'FALLO'}")
    print(f"  Ollama:        {'OK' if ollama_ok else 'FALLO'}")
    print(f"  Motor RAG:     {'OK' if rag_ok else 'FALLO'}")
    print(f"  Conversación:  {'OK' if chat_ok else 'FALLO (necesita Ollama)'}")

    print()

    if ollama_ok and rag_ok and chat_ok:
        print("  Todo listo. Ejecuta 'python main.py' para hablar con Aitana.")

    elif not ollama_ok:
        print("  Inicia Ollama y descarga el modelo antes de usar Aitana.")

    print()


if __name__ == "__main__":
    main()

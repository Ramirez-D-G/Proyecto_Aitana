"""
rag_engine.py - Motor de Retrieval-Augmented Generation (RAG) para Aitana.

Este módulo gestiona la memoria a largo plazo de Aitana usando ChromaDB
como base de datos vectorial y sentence-transformers para los embeddings.
Permite alimentar documentos, dividirlos en chunks y buscar contexto relevante.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Optional


# Ruta donde se almacena la base de datos vectorial
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Modelo de embeddings (ligero y funciona bien en español)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Nombre de la colección en ChromaDB
COLLECTION_NAME = "aitana_memoria"

# Tamaño de cada chunk de texto (en caracteres)
CHUNK_SIZE = 500

# Solapamiento entre chunks para no perder contexto
CHUNK_OVERLAP = 50

# Número de resultados relevantes a devolver en cada búsqueda
TOP_K = 3


class RAGEngine:
    """Motor RAG que gestiona la memoria a largo plazo de Aitana."""

    def __init__(self, data_dir: str = DATA_DIR):
        """
        Inicializa el motor RAG.

        Args:
            data_dir: Directorio donde se almacena la base de datos vectorial.
        """
        os.makedirs(data_dir, exist_ok=True)

        # Crear cliente persistente de ChromaDB
        self.client = chromadb.PersistentClient(path=data_dir)

        # Función de embeddings usando sentence-transformers
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        # Obtener o crear la colección
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"description": "Memoria a largo plazo de Aitana"}
        )

        print(f"[RAG] Motor inicializado. Documentos en memoria: {self.collection.count()}")

    def _split_text(self, text: str) -> List[str]:
        """
        Divide un texto largo en chunks más pequeños con solapamiento.

        Args:
            text: Texto completo a dividir.

        Returns:
            Lista de chunks de texto.
        """
        chunks = []
        start = 0
        text = text.strip()

        while start < len(text):
            end = start + CHUNK_SIZE

            # Si no estamos al final, intentar cortar en un punto natural
            if end < len(text):
                # Buscar el último salto de línea o punto dentro del chunk
                last_newline = text.rfind('\n', start, end)
                last_period = text.rfind('. ', start, end)
                cut_point = max(last_newline, last_period)

                if cut_point > start:
                    end = cut_point + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Avanzar con solapamiento
            start = end - CHUNK_OVERLAP if end < len(text) else end

        return chunks

    def add_document(self, text: str, source: str = "usuario") -> int:
        """
        Agrega un documento a la memoria a largo plazo.
        El texto se divide en chunks y cada uno se almacena con su embedding.

        Args:
            text: Contenido del documento.
            source: Nombre o identificador del origen del documento.

        Returns:
            Número de chunks añadidos.
        """
        chunks = self._split_text(text)

        if not chunks:
            print("[RAG] El documento está vacío, no se añadió nada.")
            return 0

        # Generar IDs únicos para cada chunk
        existing_count = self.collection.count()
        ids = [f"{source}_{existing_count + i}" for i in range(len(chunks))]

        # Metadatos para cada chunk
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

        # Añadir a la colección
        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

        print(f"[RAG] Añadidos {len(chunks)} chunks del documento '{source}'.")
        return len(chunks)

    def add_file(self, filepath: str) -> int:
        """
        Lee un archivo de texto y lo agrega a la memoria.

        Args:
            filepath: Ruta al archivo de texto.

        Returns:
            Número de chunks añadidos.
        """
        if not os.path.exists(filepath):
            print(f"[RAG] Error: El archivo '{filepath}' no existe.")
            return 0

        # Intentar leer con diferentes encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"[RAG] Error: No se pudo leer el archivo '{filepath}'.")
            return 0

        source = os.path.basename(filepath)
        return self.add_document(text, source=source)

    def search(self, query: str, top_k: int = TOP_K) -> List[str]:
        """
        Busca los chunks más relevantes para una consulta dada.

        Args:
            query: Texto de búsqueda (normalmente la pregunta del usuario).
            top_k: Número máximo de resultados a devolver.

        Returns:
            Lista de textos relevantes encontrados.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )

        # Extraer los documentos de los resultados
        documents = results.get("documents", [[]])[0]
        return documents

    def get_context(self, query: str) -> Optional[str]:
        """
        Obtiene contexto relevante formateado para incluir en el prompt.

        Args:
            query: Pregunta o mensaje del usuario.

        Returns:
            Texto con el contexto relevante, o None si no hay nada relevante.
        """
        results = self.search(query)

        if not results:
            return None

        context = "== CONTEXTO DE TU MEMORIA A LARGO PLAZO ==\n"
        context += "(Recuerdas esta información de conversaciones/documentos anteriores)\n\n"
        for i, doc in enumerate(results, 1):
            context += f"Recuerdo {i}: {doc}\n\n"

        return context

    def get_stats(self) -> dict:
        """
        Devuelve estadísticas de la base de datos.

        Returns:
            Diccionario con estadísticas.
        """
        return {
            "total_chunks": self.collection.count(),
            "data_dir": DATA_DIR,
            "embedding_model": EMBEDDING_MODEL,
            "collection_name": COLLECTION_NAME
        }

    def clear(self):
        """Elimina todos los documentos de la memoria a largo plazo."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )
        print("[RAG] Memoria a largo plazo borrada.")


# Si se ejecuta directamente, mostrar estadísticas
if __name__ == "__main__":
    engine = RAGEngine()
    stats = engine.get_stats()
    print("\n--- Estadísticas del motor RAG ---")
    for key, value in stats.items():
        print(f"  {key}: {value}")

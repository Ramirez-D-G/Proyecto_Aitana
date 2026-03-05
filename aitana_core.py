"""
aitana_core.py - Núcleo de Aitana, tu amiga virtual.

Este módulo conecta con Ollama para generar respuestas usando Llama,
gestiona el historial de conversación (memoria a corto plazo),
integra el motor RAG (memoria a largo plazo) y construye el prompt
con la personalidad de Aitana.
"""

import os
import json
import requests
from typing import List, Dict, Optional
from rag_engine import RAGEngine


# Configuración de Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b-instruct-q4_0"

# Memoria a corto plazo: número máximo de mensajes en el historial
MAX_HISTORY = 20  # 10 pares de usuario-Aitana

# Archivo donde se guarda el historial entre sesiones
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "historial.json")

# Archivo con la personalidad de Aitana
PERSONALITY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personalidad_aitana.txt")


def load_personality() -> str:
    """
    Carga la personalidad de Aitana desde el archivo de texto.

    Returns:
        Texto con la definición de personalidad para el system prompt.
    """
    if not os.path.exists(PERSONALITY_FILE):
        print(f"[AVISO] No se encontró '{PERSONALITY_FILE}'. Usando personalidad básica.")
        return "Eres Aitana, una amiga virtual cariñosa y cercana. Responde siempre en español."

    with open(PERSONALITY_FILE, 'r', encoding='utf-8') as f:
        return f.read()


class AitanaCore:
    """Núcleo principal de Aitana - gestiona conversación, memoria y personalidad."""

    def __init__(self, model: str = DEFAULT_MODEL, use_rag: bool = True):
        """
        Inicializa a Aitana.

        Args:
            model: Nombre del modelo en Ollama (ej: "llama3.1:8b-instruct-q4_0").
            use_rag: Si True, activa la memoria a largo plazo (RAG).
        """
        self.model = model
        self.use_rag = use_rag
        self.personality = load_personality()
        self.history: List[Dict[str, str]] = []

        # Inicializar RAG si está activado
        self.rag = None
        if use_rag:
            try:
                self.rag = RAGEngine()
            except Exception as e:
                print(f"[AVISO] No se pudo inicializar RAG: {e}")
                print("[AVISO] Aitana funcionará sin memoria a largo plazo.")
                self.use_rag = False

        # Cargar historial previo si existe
        self._load_history()

        print(f"[Aitana] Inicializada con modelo: {self.model}")
        print(f"[Aitana] Memoria a largo plazo (RAG): {'activada' if self.use_rag else 'desactivada'}")
        print(f"[Aitana] Historial cargado: {len(self.history)} mensajes")

    def _load_history(self):
        """Carga el historial de conversación desde archivo."""
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                # Mantener solo los últimos MAX_HISTORY mensajes
                if len(self.history) > MAX_HISTORY:
                    self.history = self.history[-MAX_HISTORY:]
            except (json.JSONDecodeError, IOError):
                self.history = []

    def _save_history(self):
        """Guarda el historial de conversación a archivo."""
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def _build_messages(self, user_message: str) -> List[Dict[str, str]]:
        """
        Construye la lista de mensajes para enviar a Ollama,
        incluyendo el system prompt con personalidad y contexto RAG.

        Args:
            user_message: Mensaje actual del usuario.

        Returns:
            Lista de mensajes formateada para la API de Ollama.
        """
        # Construir el system prompt
        system_prompt = self.personality

        # Agregar contexto RAG si está disponible
        if self.use_rag and self.rag:
            rag_context = self.rag.get_context(user_message)
            if rag_context:
                system_prompt += f"\n\n{rag_context}"

        # Construir la lista de mensajes
        messages = [{"role": "system", "content": system_prompt}]

        # Agregar historial de conversación
        messages.extend(self.history)

        # Agregar el mensaje actual del usuario
        messages.append({"role": "user", "content": user_message})

        return messages

    def chat(self, user_message: str) -> str:
        """
        Envía un mensaje a Aitana y obtiene su respuesta.

        Args:
            user_message: Mensaje del usuario.

        Returns:
            Respuesta de Aitana.
        """
        # Construir los mensajes con contexto completo
        messages = self._build_messages(user_message)

        try:
            # Llamar a la API de Ollama
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "num_predict": 512,
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            data = response.json()

            # Extraer la respuesta
            aitana_response = data.get("message", {}).get("content", "").strip()

            if not aitana_response:
                aitana_response = "Hmm, se me fue la onda un momento. ¿Me repites eso? 😅"

        except requests.ConnectionError:
            return (
                "[Error] No puedo conectar con Ollama. "
                "Asegúrate de que Ollama esté corriendo (ejecuta 'ollama serve' en otra terminal)."
            )
        except requests.Timeout:
            return (
                "[Error] La respuesta tardó demasiado. "
                "Puede que el modelo sea muy grande para tu hardware."
            )
        except Exception as e:
            return f"[Error] Algo salió mal: {e}"

        # Actualizar historial (memoria a corto plazo)
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": aitana_response})

        # Mantener el historial dentro del límite
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

        # Guardar historial a disco
        self._save_history()

        return aitana_response

    def chat_stream(self, user_message: str):
        """
        Envía un mensaje a Aitana y obtiene la respuesta en streaming.
        Útil para la interfaz web donde queremos ver la respuesta en tiempo real.

        Args:
            user_message: Mensaje del usuario.

        Yields:
            Fragmentos de texto de la respuesta de Aitana.
        """
        messages = self._build_messages(user_message)

        full_response = ""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "num_predict": 512,
                    }
                },
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        full_response += token
                        yield token

                    # Si es el último mensaje, terminar
                    if data.get("done", False):
                        break

        except requests.ConnectionError:
            error = (
                "[Error] No puedo conectar con Ollama. "
                "Asegúrate de que esté corriendo."
            )
            yield error
            full_response = error
        except Exception as e:
            error = f"[Error] Algo salió mal: {e}"
            yield error
            full_response = error

        # Actualizar historial
        if full_response and not full_response.startswith("[Error]"):
            self.history.append({"role": "user", "content": user_message})
            self.history.append({"role": "assistant", "content": full_response})

            if len(self.history) > MAX_HISTORY:
                self.history = self.history[-MAX_HISTORY:]

            self._save_history()

    def add_memory(self, text: str, source: str = "conversación") -> bool:
        """
        Agrega información a la memoria a largo plazo de Aitana.

        Args:
            text: Texto a recordar.
            source: Origen de la información.

        Returns:
            True si se añadió correctamente.
        """
        if not self.use_rag or not self.rag:
            print("[AVISO] RAG no está activado.")
            return False

        chunks = self.rag.add_document(text, source)
        return chunks > 0

    def add_memory_file(self, filepath: str) -> bool:
        """
        Agrega un archivo a la memoria a largo plazo.

        Args:
            filepath: Ruta al archivo de texto.

        Returns:
            True si se añadió correctamente.
        """
        if not self.use_rag or not self.rag:
            print("[AVISO] RAG no está activado.")
            return False

        chunks = self.rag.add_file(filepath)
        return chunks > 0

    def clear_history(self):
        """Borra el historial de conversación (memoria a corto plazo)."""
        self.history = []
        self._save_history()
        print("[Aitana] Historial de conversación borrado.")

    def clear_long_memory(self):
        """Borra la memoria a largo plazo."""
        if self.rag:
            self.rag.clear()

    def get_status(self) -> dict:
        """Devuelve el estado actual de Aitana."""
        status = {
            "modelo": self.model,
            "mensajes_en_historial": len(self.history),
            "rag_activado": self.use_rag,
        }
        if self.rag:
            status["documentos_en_memoria"] = self.rag.get_stats()["total_chunks"]
        return status


# Prueba rápida si se ejecuta directamente
if __name__ == "__main__":
    aitana = AitanaCore()
    print("\n--- Estado de Aitana ---")
    for key, value in aitana.get_status().items():
        print(f"  {key}: {value}")
    print("\nPrueba de conexión con Ollama...")
    response = aitana.chat("Hola, ¿cómo estás?")
    print(f"\nAitana: {response}")

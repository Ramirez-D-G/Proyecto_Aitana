# Aitana - Tu Amiga Virtual

Aitana es una amiga virtual que corre 100% localmente en tu computadora usando el modelo Llama a través de Ollama. Interactúas con ella por texto y tiene memoria a corto y largo plazo.

## Requisitos previos

- **Python 3.10+** instalado
- **Ollama** instalado y funcionando ([ollama.com](https://ollama.com))
- **GPU recomendada**: NVIDIA con 6GB+ VRAM (ej: RTX 4050)

## Instalación

### 1. Descargar el modelo de Llama en Ollama

```bash
ollama pull llama3.1:8b-instruct-q4_0
```

Si ese modelo no está disponible, puedes usar cualquier variante de Llama 3:

```bash
ollama pull llama3.1:8b
```

### 2. Crear entorno virtual e instalar dependencias

```bash
cd Proyecto_Aitana
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 3. Verificar la instalación

```bash
python test_aitana.py
```

## Uso

### Asegúrate de que Ollama esté corriendo

Ollama normalmente corre como servicio. Si no, inícialo:

```bash
ollama serve
```

### Opción A: Interfaz por consola

```bash
python main.py
```

Opciones:
- `--modelo <nombre>`: Usar un modelo diferente (ej: `--modelo llama3.1:8b`)
- `--sin-rag`: Desactivar la memoria a largo plazo

### Opción B: Interfaz web (Gradio)

```bash
python web_ui.py
```

Se abrirá en `http://localhost:7860`. Opciones:
- `--puerto <num>`: Cambiar el puerto
- `--modelo <nombre>`: Usar un modelo diferente
- `--sin-rag`: Desactivar la memoria a largo plazo

## Comandos dentro del chat

| Comando | Descripción |
|---------|-------------|
| `/salir` | Terminar la conversación |
| `/estado` | Ver estado de Aitana |
| `/limpiar` | Borrar historial de conversación |
| `/memoria <texto>` | Guardar información en memoria a largo plazo |
| `/archivo <ruta>` | Agregar archivo a la memoria |
| `/ayuda` | Ver comandos disponibles |

## Memoria a largo plazo (RAG)

Puedes alimentar documentos para que Aitana los recuerde entre sesiones:

```bash
# Agregar un archivo
python feed_documents.py mi_diario.txt

# Agregar varios archivos
python feed_documents.py archivo1.txt archivo2.txt

# Agregar una carpeta completa
python feed_documents.py --carpeta mis_documentos/

# Agregar texto directo
python feed_documents.py --texto "Mi cumpleaños es el 15 de marzo"

# Ver estadísticas
python feed_documents.py --stats

# Borrar toda la memoria
python feed_documents.py --clear
```

## Estructura del proyecto

```
Proyecto_Aitana/
├── main.py                 # Interfaz de consola
├── web_ui.py               # Interfaz web (Gradio)
├── aitana_core.py           # Núcleo: conexión con Ollama, historial, RAG
├── rag_engine.py            # Motor RAG: ChromaDB + embeddings
├── personalidad_aitana.txt  # Prompt de sistema con la personalidad
├── feed_documents.py        # Script para alimentar documentos
├── test_aitana.py           # Script de pruebas
├── requirements.txt         # Dependencias Python
├── README.md                # Este archivo
└── data/                    # Base de datos vectorial e historial (se crea automáticamente)
```

## Personalización

Edita `personalidad_aitana.txt` para modificar la personalidad de Aitana. Los cambios se aplican al reiniciar.

## Solución de problemas

- **"No puedo conectar con Ollama"**: Asegúrate de que Ollama esté corriendo (`ollama serve`).
- **Respuestas lentas**: El modelo cuantizado (`q4_0`) es el más rápido. Si usas uno más grande, será más lento.
- **Error de VRAM**: Usa un modelo más pequeño o cuantizado. La variante `q4_0` cabe en 6GB de VRAM.
- **Error de sentence-transformers**: La primera vez descarga el modelo de embeddings (~100MB). Necesitas conexión a internet solo esa primera vez.

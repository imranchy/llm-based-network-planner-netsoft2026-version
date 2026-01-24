# NetSoft 2026 – Ollama Network Assistant

## Quick start

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running and pull at least one model:
```bash
ollama pull qwen2.5:14b-instruct
ollama pull llama3.1:8b
ollama pull mistral-nemo:12b
ollama pull gemma2:9b
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Notes

- The app is **Ollama-based** (local `ollama` server on `http://127.0.0.1:11434`).
- The LLM is used primarily to **interpret the question** (intent + parameters) and the Python layer executes the graph/equipment computations deterministically.
- To force running without Ollama (pure deterministic fallback), set:
```bash
export OLLAMA_DISABLE=1
```

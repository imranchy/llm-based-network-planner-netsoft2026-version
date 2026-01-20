# NetSoft 2026 – LLM-first Network Topology Assistant

This project compares multiple open-source LLMs (via **Ollama**) on network-topology tasks.

## Key design
- **LLM-first analytics:** the model computes graph metrics (shortest/longest path, hop count, degree, etc.) from a topology snapshot.
- **No Python function-call generation:** the model must return **one strict JSON object**.
- **QoT + equipment logic remains in Python** (reference implementation) and can be injected via RAG.

## Files expected in project root
- `distance.csv`
- `city.csv`
- `questions.yaml`

## Setup
1. Create/activate your Python environment.
2. Install deps:

```bash
pip install -r requirements.txt
```

3. Install and run Ollama, and pull at least one model, e.g.:

```bash
ollama pull llama3:8b
```

## Quick CLI test

```bash
python llm_test.py
```

## Streamlit app

```bash
streamlit run app.py
```

## Benchmark
Use the Streamlit sidebar "Benchmark" section or run programmatically:

```python
import llm_agent
models = ["llama3:8b", "mistral:7b"]
df = llm_agent.run_experiment_from_yaml(models, yaml_file="questions.yaml", rag_enabled=False)
print(df.head())
```

## RAG
If you create a folder `rag_corpus/` in the project root with `.txt`, `.md`, or `.csv` files, the agent will include them when RAG is enabled.

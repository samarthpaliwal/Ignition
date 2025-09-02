# Ignition 🔥  
_A visual AI debugging tool for open-source LLMs_

---

## 📌 About  
**Ignition** is a first-of-its-kind debugging and visualization tool for AI.  
It lets you **trace, visualize, and debug** every step of an AI’s reasoning — turning black-box models into transparent, controllable systems.  

Built with **PySide6 + llama.cpp + Ollama**, it supports both **local GGUF models** and **Ollama-managed models**, with side-by-side comparison.  

---

## 🚀 Features
- ⚡ **Visual Graphs** — See every reasoning step as nodes and edges.  
- 📝 **Editable Notes** — Edit intermediate steps and re-run downstream.  
- 🔎 **Why Heatmap** — Highlights input words most relevant to the final answer.  
- 📊 **A/B Benchmark** — Compare two models on latency, tokens/sec, and output similarity.  
- 🛡️ **Guardrails Slider** — Control how cautious the model should be.  
- 🔄 **Replay Mode** — Watch the AI regenerate its final output token by token.  
- 💾 **Save/Load Sessions** — Persist and reload your debugging workflow.  

---

## ⚙️ Requirements
- Python 3.10+  
- macOS, Linux, or Windows  
- [Ollama](https://ollama.ai) (optional, for streaming models)  

Install dependencies:  
```bash
pip install -r requirements.txt

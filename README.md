
## 🔥 Ignition

Ever wonder why your AI doesn’t give you the answer you need?
**Ignition** is a first-of-its-kind visual debugging tool for AI and LLMs.
It helps ML professionals trace and visualize every step of an AI’s reasoning, turning black-box models into transparent, editable flows for debugging, analysis, and improvement.

-----

## ✨ Features

  - 📊 **Visual Trace Graph** — see how your AI reaches an answer
  - 🛠️ **Debugging Tools** — edit steps, re-run from any point, and compare outputs
  - 🔍 **Why Heatmap** — understand which tokens influenced the result
  - ⚡ **Local or Remote Models** — run with `llama.cpp`, Hugging Face `.gguf` models, or Ollama
  - 🖥️ **Modern UI** — built with PySide6 (Qt)

-----

## 🚀 Getting Started

1.  Clone the repo

<!-- end list -->

```bash
git clone https://github.com/samarthpaliwal/Ignition.git
cd Ignition
```

2.  Install dependencies

Make sure you have **Python 3.9+** installed.

```bash
pip install -r requirements.txt
```

3.  Run Ignition

<!-- end list -->

```bash
python ignition.py
```

-----

## 🔧 Built With

  * [Python 3.9+](https://www.python.org/)
  * [PySide6](https://doc.qt.io/qtforpython/) (Qt GUI)
  * [llama.cpp](https://github.com/ggerganov/llama.cpp) (local LLM inference)
  * [Ollama](https://ollama.ai) (optional model backend)
  * [Hugging Face](https://huggingface.co) (model downloads)

-----

## 📥 Download Models

Ignition is designed to work with **OpenAI’s GPT-OSS models**.
You’ll need to download them separately:

  * **GPT-OSS-120B**
    👉 [Download here](https://huggingface.co/openai/gpt-oss-120b?utm_source=chatgpt.com)

  * **GPT-OSS-20B**
    👉 [Download here](https://huggingface.co/openai/gpt-oss-20b?utm_source=chatgpt.com)

📂 Place the `.gguf` files into a folder of your choice and load them from the **Model Panel** in Ignition.
Alternatively, you can run them via **Ollama**:

```bash
ollama run <model>
```

-----

## 🧑‍💻 Contributing

Pull requests are welcome\! If you’d like to contribute, fork the repo and submit a PR.

-----

## 📄 License

MIT License © 2025 Ignition Team

-----

## ⚡ This README includes:

  - Intro + tagline
  - Features
  - Install & usage steps
  - Built With section
  - Model download instructions
  - Contribution & license

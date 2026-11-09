# Ignition

Ignition is a desktop application, written in a single Python file with PySide6, for comparing how two local language models work through the same question. Each side of the window hosts a model panel: you load a GGUF model through llama-cpp-python, point the panel at an Ollama model, or tick Mock to use canned output. A run asks the model for a numbered plan, then one short note per step, then streams a final answer. The steps are drawn as a graph you can zoom, click, edit, and re-run from, so the effect of changing an intermediate step on the final answer can be inspected side by side. The window title in the code is "Cognitive Sandbox - Local Visual Tracer".

## Requirements

- Python 3.9 or newer (the committed virtual environment was built with 3.12).
- Python packages: `PySide6`, `requests`; optionally `llama-cpp-python` for loading `.gguf` files directly and `psutil` for memory statistics in the bench. There is no `requirements.txt` in the repository; the list above comes from the imports in `Ignition.py`.
- One of: a `.gguf` model file on disk (for example a GGUF build of OpenAI's gpt-oss-20b or gpt-oss-120b), or Ollama running at `http://localhost:11434` with a model pulled, or nothing at all if you only use Mock mode.
- A desktop session; the app opens a Qt window.

## Setup

```bash
git clone https://github.com/samarthpaliwal/Ignition.git
cd Ignition
python3 -m venv .venv
source .venv/bin/activate
pip install PySide6 requests llama-cpp-python psutil
```

If you intend to use Ollama instead of a local GGUF file, `llama-cpp-python` can be skipped:

```bash
ollama pull mistral
ollama serve
```

Model weights are not part of this repository. The `gpt-oss-120b`, `gpt-oss-20b`, and `llama.cpp` entries in the tree are dangling submodule pointers with no `.gitmodules`; download weights separately from Hugging Face and keep them outside the repository or in an ignored folder.

## Usage

```bash
python Ignition.py
```

In each model panel:

- Set `Alias` (defaults are `gpt-oss-120b` on the left and `gpt-oss-20b` on the right).
- Either click `Browse...` and pick a `.gguf` file, adjust `ctx`, `threads`, and `gpu layers`, then click `Load`; or tick `Use Ollama`, type a model name (default `mistral`), and click `Load`; or tick `Mock` and click `Load`.
- Move the `Guardrail` slider to change the safety preamble prepended to every prompt. Paste a URL or snippet into `Source` to have it appended as grounding text.

Type a question in `Query / Task` and click `Run Both`. Each panel shows the plan graph, the per-step notes in a list, the streamed final answer, and a "Why Heatmap" that highlights query and note tokens that also appear in the answer.

Per-panel buttons: `Edit Selected Node...` (or double-click a node) changes a step's note and marks it edited; `Re-run from Here` keeps edited notes and regenerates everything downstream of the selected step; `Export Graph PNG...` saves the graph; `Replay` re-streams the last answer; `Explain impact of this edit` shows a unified diff between the previous and current final answers. Mouse wheel zooms the graph; the `Fit`, `100%`, `+`, and `-` toolbar actions adjust zoom.

Main toolbar: `Save Session` and `Load Session` write and read a JSON file containing both panels' configuration, traces, and answers; `Edit Prompts` opens the five prompt templates for editing; `Run A/B Bench` runs five built-in questions through both panels and writes `bench_results.json` to the working directory; `About` shows a summary.

## How it works

`LocalLLM` wraps the three backends behind two methods: `chat_json`, which sends a system and user prompt and parses the reply as JSON (stripping code fences), and `chat_stream_final`, which yields the final answer token by token. For llama.cpp it calls `Llama.create_chat_completion`; for Ollama it posts to `/api/generate`; Mock returns fixed strings. `TraceWorker` is a `QThread` that runs the three-stage pipeline (plan, per-step notes, final answer) and emits Qt signals as each piece arrives. When re-running from a step it reuses the existing plan and any notes marked edited and only regenerates later steps. `GraphView` is a `QGraphicsView` that lays the steps out as rounded nodes connected by lines, highlighting edited nodes. The heatmap is a token-overlap proxy computed in `score_proxy_attention`, not real attention from the model. `ModelPanel` owns one model, one trace, and the controls; `MainWindow` holds two panels, the query box, session save and load, the prompt editor, and the bench runner.

## Project layout

```
Ignition.py        the whole application: theme, data models, LLM wrapper,
                   worker thread, graph view, model panel, main window
README.md          this file
.gitattributes     Git LFS rules for model and media files
.gitignore         ignore rules (see below)
.vscode/           editor settings
```

The committed `.venv/` directory and the `gpt-oss-120b`, `gpt-oss-20b`, and `llama.cpp` submodule pointers should be removed from the tree; they are covered by `.gitignore`.

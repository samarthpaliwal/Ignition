from __future__ import annotations
import os
import re
import sys
import json
import time
import requests
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from difflib import unified_diff

try:
    import psutil  # optional; used for memory stats in bench
except Exception:
    psutil = None

from PySide6.QtCore import Qt, QThread, Signal, QPointF, QSize
from PySide6.QtGui import (
    QBrush, QPen, QColor, QPainterPath, QAction, QPainter, QPalette, QIcon
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit,
    QGraphicsView, QGraphicsScene, QGraphicsTextItem, QGraphicsPathItem,
    QListWidget, QListWidgetItem, QSplitter, QMessageBox,
    QInputDialog, QSpinBox, QCheckBox, QToolBar, QGraphicsDropShadowEffect,
    QProgressBar, QDialog, QDialogButtonBox, QFormLayout
)

# Try to import llama.cpp (local inference)
try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover
    Llama = None

# ============================= THEME ============================= #


def apply_theme(app: QApplication, dark: bool = True) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    if dark:
        bg = QColor(26, 28, 33)
        card = QColor(36, 39, 46)
        text = QColor(232, 235, 241)
        sub = QColor(140, 147, 160)
        acc = QColor(64, 132, 247)
        pal.setColor(QPalette.Window, bg)
        pal.setColor(QPalette.Base, card)
        pal.setColor(QPalette.AlternateBase, bg)
        pal.setColor(QPalette.Button, card)
        pal.setColor(QPalette.Text, text)
        pal.setColor(QPalette.ButtonText, text)
        pal.setColor(QPalette.BrightText, text)
        pal.setColor(QPalette.WindowText, text)
        pal.setColor(QPalette.PlaceholderText, sub)
        pal.setColor(QPalette.Highlight, acc)
        pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    else:
        pal = app.palette()
    app.setPalette(pal)
    app.setStyleSheet("""
    * { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    QMainWindow { background: #1a1c21; }
    QTextEdit, QLineEdit, QListWidget {
        border: 1px solid #2b2f36; border-radius: 10px; padding: 8px; background: #24272e; color: #e8ebf1;
    }
    QLabel { color: #c9ced9; }
    QPushButton {
        border-radius: 10px; padding: 8px 14px; border: 1px solid #2b2f36; background: #2b2f36; color: #e8ebf1;
    }
    QPushButton:hover { background: #323844; }
    QPushButton:pressed { background: #2a2f3a; }
    QPushButton#primary {
        background: #3a7bfd; border-color: #3a7bfd; color: white; font-weight: 600;
    }
    QPushButton#primary:hover { background: #3570e6; }
    QToolBar { background: #20232a; border-bottom: 1px solid #2b2f36; spacing: 8px; padding: 6px; }
    QSplitter::handle { background: #1f232b; width: 6px; }
    """)

# ============================= HELPERS ============================= #


def diff_text(a: str, b: str) -> str:
    if a is None:
        a = ""
    if b is None:
        b = ""
    lines = list(unified_diff(a.splitlines(True), b.splitlines(
        True), fromfile="before", tofile="after"))
    return "".join(lines) if lines else "(no change)"


def safety_prefix(level: int) -> str:
    if level <= 2:
        return "Answer directly. Avoid revealing chain-of-thought."
    if level <= 6:
        return "Be cautious. Decline unsafe content, avoid chain-of-thought, and keep to high-level notes only."
    return "Be very cautious. Decline unsafe/unknown. Provide high-level, non-sensitive responses. Never reveal chain-of-thought."


def tokenize_simple(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def score_proxy_attention(query: str, notes: List[str], final_answer: str) -> Dict[str, float]:
    fa_tokens = tokenize_simple(final_answer)
    base: Dict[str, float] = {}
    for t in fa_tokens:
        base[t] = base.get(t, 0) + 1
    if base:
        mx = max(base.values())
        for k in list(base.keys()):
            base[k] = base[k] / mx
    scores: Dict[str, float] = {}
    for token in tokenize_simple(query + " " + " ".join(notes)):
        scores[token] = max(scores.get(token, 0.0), base.get(token, 0.0))
    return scores

# ============================= DATA MODELS ============================= #


@dataclass
class StepNode:
    idx: int
    title: str
    note: str = ""
    edited: bool = False


@dataclass
class TraceGraph:
    steps: List[StepNode] = field(default_factory=list)
    final_answer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [{"idx": s.idx, "title": s.title, "note": s.note, "edited": s.edited} for s in self.steps],
            "final_answer": self.final_answer,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceGraph":
        tg = cls()
        for sd in data.get("steps", []):
            tg.steps.append(StepNode(idx=sd.get("idx", 0), title=sd.get(
                "title", ""), note=sd.get("note", ""), edited=sd.get("edited", False)))
        tg.final_answer = data.get("final_answer", "")
        return tg


@dataclass
class ModelConfig:
    alias: str = "gpt-oss-120b"
    model_path: str = ""
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0
    use_mock: bool = False
    use_ollama: bool = False
    ollama_model: str = ""  # e.g. "mistral" or "llama3"

# ============================= LLM WRAPPER ============================= #


class LocalLLM:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.llm: Optional[Any] = None
        self.lock = threading.Lock()
        self._guard_level = 3
        self._source_text = ""

    def load(self):
        if self.cfg.use_mock or self.cfg.use_ollama:
            return
        if Llama is None:
            raise RuntimeError(
                "llama_cpp is not installed. Run: pip install llama-cpp-python")
        with self.lock:
            self.llm = Llama(
                model_path=self.cfg.model_path,
                n_ctx=self.cfg.n_ctx,
                n_threads=self.cfg.n_threads,
                n_gpu_layers=self.cfg.n_gpu_layers,
                verbose=False,
            )

    # Ollama
    def _ollama_generate_json(self, prompt: str, temperature: float, max_tokens: int) -> dict:
        payload = {"model": self.cfg.ollama_model, "prompt": prompt, "options": {
            "temperature": temperature, "num_predict": max_tokens}, "stream": False}
        try:
            r = requests.post(
                "http://localhost:11434/api/generate", json=payload, timeout=600)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                "Ollama request failed. Is ollama running and the model pulled?") from e
        return r.json()

    def _ollama_stream(self, prompt: str, temperature: float, max_tokens: int):
        payload = {"model": self.cfg.ollama_model, "prompt": prompt, "options": {
            "temperature": temperature, "num_predict": max_tokens}, "stream": True}
        try:
            with requests.post("http://localhost:11434/api/generate", json=payload, stream=True, timeout=600) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "response" in obj:
                            for ch in obj["response"]:
                                yield ch
                        if obj.get("done"):
                            break
                    except Exception:
                        continue
        except Exception as e:
            raise RuntimeError(
                "Ollama stream failed. Start ollama and pull the model.") from e

    # JSON helper
    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> Dict[str, Any]:
        guard = safety_prefix(int(getattr(self, "_guard_level", 3)))
        src = getattr(self, "_source_text", "")
        up = user_prompt
        if src:
            up += f"\n\nUse this source for grounding if relevant:\n{src}\n\nCite it briefly in the final line."

        if self.cfg.use_mock:
            if "Produce a numbered plan" in up:
                return {"steps": [
                    {"id": 1, "title": "Fact retrieval"},
                    {"id": 2, "title": "Context mapping"},
                    {"id": 3, "title": "Reasoning"},
                    {"id": 4, "title": "Synthesis"},
                ]}
            if "provide a SINGLE sentence" in up or "provide a single sentence" in up:
                sid = 1
                for line in up.splitlines():
                    if line.strip().lower().startswith("step id"):
                        try:
                            sid = int(line.split(":")[-1].strip())
                        except:
                            pass
                return {"id": sid, "note": "Brief, high-level description of this step without internal reasoning."}
            return {}

        if self.cfg.use_ollama:
            prompt = f"{guard}\n\n{system_prompt}\n\n{up}"
            res = self._ollama_generate_json(prompt, temperature, max_tokens)
            text = res.get("response", "").strip()
            try:
                cleaned = text
                if cleaned.startswith("```"):
                    cleaned = "\n".join(
                        [ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")])
                return json.loads(cleaned)
            except Exception:
                return {}

        with self.lock:
            messages = [
                {"role": "system", "content": guard + "\n" + system_prompt},
                {"role": "user", "content": up},
            ]
            out = self.llm.create_chat_completion(
                messages=messages, temperature=temperature, max_tokens=max_tokens, stop=["```", "\n\n\n"])
        text = out["choices"][0]["message"]["content"] if out and out.get(
            "choices") else ""
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(
                    [ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")])
            return json.loads(cleaned)
        except Exception:
            return {}

    # Final stream
    def chat_stream_final(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 1024):
        guard = safety_prefix(int(getattr(self, "_guard_level", 3)))
        src = getattr(self, "_source_text", "")
        up = user_prompt
        if src:
            up += f"\n\nUse this source for grounding if relevant:\n{src}\n\nCite it briefly in the final line."

        if self.cfg.use_mock:
            ans = "This is a concise final answer tailored to the task, produced without revealing chain-of-thought."
            for ch in ans:
                yield ch
                time.sleep(0.01)
            return

        if self.cfg.use_ollama:
            prompt = f"{guard}\n\n{system_prompt}\n\n{up}"
            for ch in self._ollama_stream(prompt, temperature, max_tokens):
                yield ch
            return

        with self.lock:
            messages = [
                {"role": "system", "content": guard + "\n" + system_prompt},
                {"role": "user", "content": up},
            ]
            stream = self.llm.create_chat_completion(
                messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        for ev in stream:
            if not ev or "choices" not in ev:
                continue
            delta = ev["choices"][0]["delta"].get("content", "")
            if delta:
                for ch in delta:
                    yield ch

# ============================= PROMPTS & BENCH ============================= #


PROMPTS = {
    "SYSTEM_PLANNER": (
        "You are a planning assistant. You must NOT reveal chain-of-thought or detailed rationales. "
        "Your job is to produce short step TITLES and brief NOTES, strictly in JSON, no extra text."
    ),
    "USER_PLAN_TEMPLATE": (
        "Task: {task}\n\n"
        "Produce a numbered plan of 3–6 steps with very short titles (<=7 words each) that describe the operation type "
        "(e.g., 'Fact retrieval', 'Context mapping', 'Reasoning', 'Synthesis', 'Verification').\n"
        "Do NOT include explanations. Return pure JSON only in this exact format:\n\n"
        "{{\n  \"steps\": [\n    {{\"id\": 1, \"title\": \"...\"}},\n    {{\"id\": 2, \"title\": \"...\"}}\n  ]\n}}\n"
    ),
    "USER_NOTE_TEMPLATE": (
        "Task: {task}\nStep ID: {sid}\nStep Title: {title}\n\n"
        "Given the task and this step, provide a SINGLE sentence (<=25 words) describing what this step will cover, at a high level, without revealing internal reasoning or specific facts.\n"
        "Return pure JSON only in this exact format: {{\"id\": {sid}, \"note\": \"...\"}}\n"
    ),
    "SYSTEM_FINAL": "You answer directly and concisely without revealing chain-of-thought.",
    "USER_FINAL_TEMPLATE": (
        "Task: {task}\n"
        "Use the following high-level step titles and brief notes (some may be user-edited constraints).\n"
        "{steps_serialized}\n\n"
        "Now provide the final answer succinctly. Do NOT show your reasoning or steps; just the answer."
    ),
}

MINI_BENCH = [
    "What is the capital of Japan?",
    "Explain the difference between TCP and UDP in one paragraph.",
    "Write a Python function to check if a number is prime.",
    "Summarize: The mitochondrion is the powerhouse of the cell.",
    "Translate to Spanish: 'Knowledge is power.'",
    "List three risks of AI systems.",
    "Given 3x + 5 = 20, solve for x.",
    "What is a binary search? One paragraph.",
    "Name two causes of climate change.",
    "Write a short haiku about the ocean.",
]

# ============================= PROMPT EDITOR ============================= #


class PromptEditor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Prompts")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.sys_plan = QTextEdit(PROMPTS["SYSTEM_PLANNER"])
        self.user_plan = QTextEdit(PROMPTS["USER_PLAN_TEMPLATE"])
        self.user_note = QTextEdit(PROMPTS["USER_NOTE_TEMPLATE"])
        self.sys_final = QTextEdit(PROMPTS["SYSTEM_FINAL"])
        self.user_final = QTextEdit(PROMPTS["USER_FINAL_TEMPLATE"])
        form.addRow("SYSTEM_PLANNER", self.sys_plan)
        form.addRow("USER_PLAN_TEMPLATE", self.user_plan)
        form.addRow("USER_NOTE_TEMPLATE", self.user_note)
        form.addRow("SYSTEM_FINAL", self.sys_final)
        form.addRow("USER_FINAL_TEMPLATE", self.user_final)
        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Save |
                                QDialogButtonBox.Cancel)
        lay.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def accept(self):
        PROMPTS["SYSTEM_PLANNER"] = self.sys_plan.toPlainText()
        PROMPTS["USER_PLAN_TEMPLATE"] = self.user_plan.toPlainText()
        PROMPTS["USER_NOTE_TEMPLATE"] = self.user_note.toPlainText()
        PROMPTS["SYSTEM_FINAL"] = self.sys_final.toPlainText()
        PROMPTS["USER_FINAL_TEMPLATE"] = self.user_final.toPlainText()
        super().accept()

# ============================= WORKER ============================= #


class TraceWorker(QThread):
    planned = Signal(list)
    step_noted = Signal(int, str)
    token = Signal(str)
    finished = Signal(str)
    error = Signal(str)
    metrics = Signal(dict)

    def __init__(self, llm: LocalLLM, task: str, existing: Optional[TraceGraph] = None, rerun_from_idx: Optional[int] = None, prompts: Optional[Dict[str, str]] = None):
        super().__init__()
        self.llm = llm
        self.task = task
        self.existing = existing
        self.rerun_from_idx = rerun_from_idx
        self.prompts = prompts or PROMPTS

    def run(self):
        try:
            t0 = time.time()
            token_count = 0
            final_buf: List[str] = []

            SYSTEM_PLANNER = self.prompts["SYSTEM_PLANNER"]
            USER_PLAN_TEMPLATE = self.prompts["USER_PLAN_TEMPLATE"]
            USER_NOTE_TEMPLATE = self.prompts["USER_NOTE_TEMPLATE"]
            SYSTEM_FINAL = self.prompts["SYSTEM_FINAL"]
            USER_FINAL_TEMPLATE = self.prompts["USER_FINAL_TEMPLATE"]

            if self.existing and self.existing.steps and self.rerun_from_idx is not None:
                steps = [StepNode(idx=s.idx, title=s.title, note=s.note,
                                  edited=s.edited) for s in self.existing.steps]
                self.planned.emit(steps)
            else:
                plan_json = self.llm.chat_json(SYSTEM_PLANNER, USER_PLAN_TEMPLATE.format(
                    task=self.task), temperature=0.1, max_tokens=384)
                steps_raw = plan_json.get("steps", [])
                if not steps_raw:
                    steps_raw = [
                        {"id": 1, "title": "Fact retrieval"},
                        {"id": 2, "title": "Reasoning"},
                        {"id": 3, "title": "Synthesis"},
                    ]
                steps = [StepNode(idx=sr.get(
                    "id", i+1), title=sr.get("title", f"Step {i+1}")) for i, sr in enumerate(steps_raw)]
                self.planned.emit(steps)

            start_i = 0 if self.rerun_from_idx is None else self.rerun_from_idx
            for i in range(start_i, len(steps)):
                s = steps[i]
                if self.existing and i < len(self.existing.steps):
                    old = self.existing.steps[i]
                    if old.edited and old.note:
                        steps[i].note = old.note
                        self.step_noted.emit(i, old.note)
                        continue
                note_json = self.llm.chat_json(SYSTEM_PLANNER, USER_NOTE_TEMPLATE.format(
                    task=self.task, sid=s.idx, title=s.title), temperature=0.2, max_tokens=256)
                note = note_json.get("note") or note_json.get(
                    "Note") or "High-level summary for this step."
                steps[i].note = note
                if self.existing and i < len(self.existing.steps):
                    self.existing.steps[i].note = note
                self.step_noted.emit(i, note)

            ser = [{"id": s.idx, "title": s.title,
                    "note": (s.note or "")} for s in steps]
            steps_serialized = json.dumps({"steps": ser}, ensure_ascii=False)
            final_user = USER_FINAL_TEMPLATE.format(
                task=self.task, steps_serialized=steps_serialized)

            for ch in self.llm.chat_stream_final(SYSTEM_FINAL, final_user, temperature=0.2, max_tokens=1024):
                token_count += 1
                final_buf.append(ch)
                self.token.emit(ch)

            self.finished.emit("done")
            self.metrics.emit({"elapsed_sec": round(
                time.time() - t0, 3), "approx_tokens": token_count, "final": "".join(final_buf)})
        except Exception as e:
            self.error.emit(str(e))

# ============================= GRAPH VIEW (with Zoom Fit) ============================= #


class GraphView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setRenderHints(self.renderHints() |
                            QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setBackgroundBrush(QBrush(QColor(26, 28, 33)))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._auto_fit = True  # fit scene on resize/redraw
        self._current_scale = 0.7

        self.nodes: List[QGraphicsPathItem] = []
        self.texts: List[QGraphicsTextItem] = []
        self.edges: List[QGraphicsPathItem] = []
        self.data: TraceGraph = TraceGraph()
        self.node_clicked_callback = None

    # zoom controls
    def zoom_reset(self):
        self._current_scale = 1.0
        self.resetTransform()
        if self._auto_fit:
            self.fitInView(self.scene.itemsBoundingRect(
            ).adjusted(-40, -40, 40, 40), Qt.KeepAspectRatio)

    def zoom_fit(self):
        self._auto_fit = True
        self.zoom_reset()

    def zoom_in(self):
        self._auto_fit = False
        self.scale(1.2, 1.2)
        self._current_scale *= 1.2

    def zoom_out(self):
        self._auto_fit = False
        self.scale(1/1.2, 1/1.2)
        self._current_scale /= 1.2

    def wheelEvent(self, event):
        # Smooth zoom at cursor; disable auto-fit when user interacts
        self._auto_fit = False
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self.scale(factor, factor)
        self._current_scale *= factor

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit and not self.scene.itemsBoundingRect().isNull():
            self.fitInView(self.scene.itemsBoundingRect(
            ).adjusted(-40, -40, 40, 40), Qt.KeepAspectRatio)

    def clear_all(self):
        self.scene.clear()
        self.nodes, self.texts, self.edges = [], [], []

    def set_graph(self, trace: TraceGraph):
        self.data = trace
        self.redraw()

    def _rounded_node(self, w, h, edited: bool) -> QGraphicsPathItem:
        path = QPainterPath()
        path.addRoundedRect(0, 0, w, h, 12, 12)
        item = QGraphicsPathItem(path)
        item.setBrush(QBrush(QColor(32, 35, 42)))
        border = QColor(64, 132, 247) if edited else QColor(55, 59, 68)
        item.setPen(QPen(border, 2))
        item.setFlag(QGraphicsPathItem.ItemIsSelectable, True)
        # (keep static layout in this version; movable nodes exist in the extended build)
        return item

    def redraw(self):
        self.clear_all()
        if not self.data.steps:
            return
        x = 40
        y = 20
        box_w, box_h = 420, 92
        v_gap = 26
        for i, s in enumerate(self.data.steps):
            node = self._rounded_node(box_w, box_h, s.edited)
            node.setPos(x, y + i * (box_h + v_gap))
            self.scene.addItem(node)
            self.nodes.append(node)

            label = f"<b>{s.idx}. {s.title}</b><br><span style='color:#8c93a0'>{(s.note or '').strip()}</span>"
            txt = QGraphicsTextItem()
            txt.setHtml(label)
            txt.setTextWidth(box_w - 20)
            txt.setDefaultTextColor(QColor(232, 235, 241))
            txt.setPos(x + 10, y + i * (box_h + v_gap) + 8)
            self.scene.addItem(txt)
            self.texts.append(txt)

        for i in range(len(self.nodes) - 1):
            a = self.nodes[i]
            b = self.nodes[i+1]
            ax = a.pos().x() + box_w
            ay = a.pos().y() + box_h/2
            bx = b.pos().x()
            by = b.pos().y() + box_h/2
            path = QPainterPath(QPointF(ax, ay))
            midx = (ax + bx) / 2
            path.cubicTo(QPointF(midx, ay), QPointF(midx, by), QPointF(bx, by))
            curve = QGraphicsPathItem(path)
            curve.setPen(QPen(QColor(80, 86, 98), 2))
            self.scene.addItem(curve)
            self.edges.append(curve)

        self.scene.setSceneRect(
            self.scene.itemsBoundingRect().adjusted(-40, -40, 40, 40))
        if self._auto_fit:
            self.fitInView(self.scene.itemsBoundingRect(
            ).adjusted(-40, -40, 40, 40), Qt.KeepAspectRatio)

    def mouseDoubleClickEvent(self, event):
        pos = self.mapToScene(event.pos())
        for idx, node in enumerate(self.nodes):
            if node.contains(node.mapFromScene(pos)):
                if callable(self.node_clicked_callback):
                    self.node_clicked_callback(idx)
                break
        super().mouseDoubleClickEvent(event)

# ============================= MODEL PANEL ============================= #


class ModelPanel(QWidget):
    run_complete = Signal(dict)

    def __init__(self, alias_default: str, get_task: callable):
        super().__init__()
        self.get_task = get_task  # << safe way to read query text
        self.cfg = ModelConfig(alias=alias_default)
        self.llm = LocalLLM(self.cfg)
        self.trace = TraceGraph()
        self.worker: Optional[TraceWorker] = None
        self._last_final = ""
        self._replay_tokens: List[str] = []

        lay = QVBoxLayout(self)

        # top toolbar for this panel: zoom
        panel_tb = QToolBar()
        act_fit = QAction("Fit", self)
        act_100 = QAction("100%", self)
        act_zoomin = QAction("+", self)
        act_zoomout = QAction("–", self)
        panel_tb.addAction(act_fit)
        panel_tb.addAction(act_100)
        panel_tb.addAction(act_zoomin)
        panel_tb.addAction(act_zoomout)
        lay.addWidget(panel_tb)

        # Config row 1
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Alias:"))
        self.alias_edit = QLineEdit(self.cfg.alias)
        self.alias_edit.setMinimumWidth(140)
        row1.addWidget(self.alias_edit, 1)
        row1.addWidget(QLabel("Model GGUF:"))
        self.model_path_edit = QLineEdit()
        row1.addWidget(self.model_path_edit, 3)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        row1.addWidget(browse_btn)
        lay.addLayout(row1)

        # Config row 2
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("ctx:"))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(512, 32768)
        self.ctx_spin.setValue(self.cfg.n_ctx)
        row2.addWidget(self.ctx_spin)
        row2.addWidget(QLabel("threads:"))
        self.th_spin = QSpinBox()
        self.th_spin.setRange(1, 64)
        self.th_spin.setValue(self.cfg.n_threads)
        row2.addWidget(self.th_spin)
        row2.addWidget(QLabel("gpu layers:"))
        self.gpu_spin = QSpinBox()
        self.gpu_spin.setRange(0, 200)
        self.gpu_spin.setValue(self.cfg.n_gpu_layers)
        row2.addWidget(self.gpu_spin)
        self.mock_chk = QCheckBox("Mock")
        row2.addWidget(self.mock_chk)
        self.ollama_chk = QCheckBox("Use Ollama")
        row2.addWidget(self.ollama_chk)
        row2.addWidget(QLabel("model:"))
        self.ollama_model_edit = QLineEdit()
        self.ollama_model_edit.setPlaceholderText("mistral")
        self.ollama_model_edit.setFixedWidth(120)
        row2.addWidget(self.ollama_model_edit)
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self._load)
        row2.addWidget(self.load_btn)
        self.status_lbl = QLabel("Not loaded")
        row2.addWidget(self.status_lbl)
        lay.addLayout(row2)

        # Graph + list
        self.graph = GraphView()
        self.graph.setMinimumHeight(280)
        self.graph.node_clicked_callback = self._edit_node_by_index
        self.list = QListWidget()
        mid_split = QSplitter(Qt.Vertical)
        mid_split.addWidget(self.graph)
        mid_split.addWidget(self.list)
        mid_split.setSizes([500, 200])

        # Final + Why
        self.final_out = QTextEdit()
        self.final_out.setReadOnly(True)
        self.why_out = QTextEdit()
        self.why_out.setReadOnly(True)
        self.why_out.setPlaceholderText("Why Heatmap: appears after a run")
        detail_split = QSplitter(Qt.Vertical)
        detail_split.addWidget(self.final_out)
        detail_split.addWidget(self.why_out)
        detail_split.setSizes([350, 150])

        main_split = QSplitter(Qt.Vertical)
        main_split.addWidget(mid_split)
        main_split.addWidget(detail_split)
        main_split.setSizes([600, 300])
        lay.addWidget(main_split, 1)

        # action row
        row3 = QHBoxLayout()
        self.edit_btn = QPushButton("Edit Selected Node…")
        self.edit_btn.clicked.connect(self._edit_selected)
        row3.addWidget(self.edit_btn)
        self.rerun_btn = QPushButton("Re-run from Here")
        self.rerun_btn.clicked.connect(self._rerun_from_selected)
        row3.addWidget(self.rerun_btn)
        self.export_btn = QPushButton("Export Graph PNG…")
        self.export_btn.clicked.connect(self._export_png)
        row3.addWidget(self.export_btn)
        self.replay_btn = QPushButton("Replay")
        self.replay_btn.clicked.connect(self._replay)
        row3.addWidget(self.replay_btn)
        lay.addLayout(row3)

        # guardrail + source + flow
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Guardrail:"))
        self.guard_slider = QSpinBox()
        self.guard_slider.setRange(0, 10)
        self.guard_slider.setValue(3)
        row4.addWidget(self.guard_slider)
        row4.addSpacing(12)
        row4.addWidget(QLabel("Source (URL/snippet):"))
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("https://… or paste short snippet")
        row4.addWidget(self.source_edit, 1)
        self.explain_btn = QPushButton("Explain impact of this edit")
        self.explain_btn.clicked.connect(self._explain_impact)
        row4.addWidget(self.explain_btn)
        self.flow_bar = QProgressBar()
        self.flow_bar.setRange(0, 1)
        self.flow_bar.setFixedWidth(140)
        row4.addWidget(self.flow_bar)
        lay.addLayout(row4)

        # wire panel zoom actions
        act_fit.triggered.connect(self.graph.zoom_fit)
        act_100.triggered.connect(self.graph.zoom_reset)
        act_zoomin.triggered.connect(self.graph.zoom_in)
        act_zoomout.triggered.connect(self.graph.zoom_out)

    # Config
    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", filter="Model (*.gguf)")
        if path:
            self.model_path_edit.setText(path)

    def _load(self):
        self.cfg.alias = self.alias_edit.text().strip() or self.cfg.alias
        self.cfg.model_path = self.model_path_edit.text().strip()
        self.cfg.n_ctx = int(self.ctx_spin.value())
        self.cfg.n_threads = int(self.th_spin.value())
        self.cfg.n_gpu_layers = int(self.gpu_spin.value())
        self.cfg.use_mock = self.mock_chk.isChecked()
        self.cfg.use_ollama = self.ollama_chk.isChecked()
        self.cfg.ollama_model = (self.ollama_model_edit.text(
        ).strip() or "mistral") if self.cfg.use_ollama else ""
        try:
            self.llm.load()
            if self.cfg.use_mock:
                self.status_lbl.setText("Loaded ✔ (Mock)")
            elif self.cfg.use_ollama:
                self.status_lbl.setText(
                    f"Loaded ✔ (Ollama: {self.cfg.ollama_model})")
            else:
                self.status_lbl.setText("Loaded ✔ (GGUF)")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            self.status_lbl.setText("Load failed ✖")

    # Run
    def run_full(self, task: str):
        if not (self.cfg.use_mock or self.cfg.use_ollama) and (not self.cfg.model_path or self.llm.llm is None):
            QMessageBox.warning(self, "Model not loaded",
                                "Please load a model or enable Mock/Ollama mode.")
            return
        self.llm._guard_level = int(self.guard_slider.value())
        self.llm._source_text = self.source_edit.text().strip()

        self.flow_bar.setRange(0, 0)  # busy
        self.final_out.clear()
        self.list.clear()
        self.trace = TraceGraph()
        self.graph.set_graph(self.trace)
        self.graph.zoom_fit()
        self._start_worker(task, existing=None, rerun_from_idx=None)

    def _start_worker(self, task: str, existing: Optional[TraceGraph], rerun_from_idx: Optional[int]):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(
                self, "Busy", "A run is already in progress.")
            return
        self.rerun_btn.setEnabled(False)
        self.edit_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.worker = TraceWorker(
            self.llm, task, existing, rerun_from_idx, prompts=PROMPTS)
        self.worker.planned.connect(self._on_planned)
        self.worker.step_noted.connect(self._on_step_noted)
        self.worker.token.connect(self._on_token)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.metrics.connect(self._on_metrics)
        self._replay_tokens = []
        self.worker.finished.connect(lambda _: self._run_done_ui())
        self.worker.error.connect(lambda _: self._run_done_ui())
        self.worker.start()

    def _run_done_ui(self):
        self.rerun_btn.setEnabled(True)
        self.edit_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.flow_bar.setRange(0, 1)
        self.graph.zoom_fit()

    def _on_planned(self, steps: List[StepNode]):
        self.trace.steps = [
            StepNode(idx=s.idx, title=s.title, note="", edited=False) for s in steps]
        self._refresh_lists_and_graph()

    def _on_step_noted(self, i: int, note: str):
        if i < len(self.trace.steps):
            edited = self.trace.steps[i].edited
            self.trace.steps[i].note = note
            self.trace.steps[i].edited = edited
            self._refresh_lists_and_graph()

    def _on_token(self, ch: str):
        self._replay_tokens.append(ch)
        self.final_out.moveCursor(self.final_out.textCursor().End)
        self.final_out.insertPlainText(ch)
        self.final_out.moveCursor(self.final_out.textCursor().End)

    def _on_finished(self, _: str):
        self.trace.final_answer = self.final_out.toPlainText()
        # Why heatmap (proxy)
        task = (self.get_task() or "")
        notes = [s.note for s in self.trace.steps]
        scores = score_proxy_attention(task, notes, self.trace.final_answer)
        self._render_heatmap(task, scores)
        # session log event
        self.run_complete.emit({
            "alias": self.cfg.alias,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "task": task,
            "steps": self.trace.to_dict()["steps"],
            "final": self.trace.final_answer,
        })

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Run Error", msg)

    def _on_metrics(self, m: dict):
        # status
        t = m.get("elapsed_sec", 0.0)
        tok = m.get("approx_tokens", 0)
        self.status_lbl.setText(f"Done ✔  {t}s, ~{tok} tokens")

        # compute diff silently (no popup)
        new_final = m.get("final", "")
        # kept for potential future logging
        _ = diff_text(self._last_final, new_final)

        # show the final answer inline above the Heatmap
        escaped = (new_final or "").replace("&", "&amp;").replace(
            "<", "&lt;").replace(">", "&gt;")
        html = (
            "<div style='font-size:14px;color:#e8ebf1;'>"
            "<div style='display:flex;align-items:center;gap:8px;'>"
            "<span style='font-weight:700;'>Final Answer</span>"
            "<span style='opacity:.7;font-size:12px'>(most recent)</span>"
            "</div>"
            f"<div style='margin-top:6px;white-space:pre-wrap;line-height:1.35;'>{escaped}</div>"
            "</div>"
        )
        self.final_out.setHtml(html)

        # remember latest for next diff
        self._last_final = new_final

    def _refresh_lists_and_graph(self):
        self.list.clear()
        for s in self.trace.steps:
            tag = " (edited)" if s.edited else ""
            self.list.addItem(QListWidgetItem(
                f"{s.idx}. {s.title}{tag}\n{(s.note or '').strip()}"))
        self.graph.set_graph(self.trace)

    # Editing
    def _edit_node_by_index(self, i: int):
        if i < 0 or i >= len(self.trace.steps):
            return
        cur = self.trace.steps[i]
        txt, ok = QInputDialog.getMultiLineText(
            self, "Edit Step Note", f"{cur.idx}. {cur.title}", cur.note)
        if ok:
            cur.note = txt.strip()
            cur.edited = True
            self._refresh_lists_and_graph()

    def _edit_selected(self):
        row = self.list.currentRow()
        self._edit_node_by_index(row)

    def _rerun_from_selected(self):
        row = self.list.currentRow()
        if row < 0:
            QMessageBox.information(
                self, "Select a step", "Select a step to re-run from.")
            return
        task = (self.get_task() or "")
        if not task:
            QMessageBox.information(
                self, "No task", "Enter a task/query first at the top.")
            return
        self.final_out.clear()
        self._start_worker(task, existing=self.trace, rerun_from_idx=row)

    # Export
    def _export_png(self):
        if self.graph.scene is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Graph", filter="PNG (*.png)")
        if not path:
            return
        rect = self.graph.scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
        img_w = max(800, int(rect.width()))
        img_h = max(400, int(rect.height()))
        from PySide6.QtGui import QImage
        img = QImage(img_w, img_h, QImage.Format_ARGB32)
        img.fill(QColor(255, 255, 255))
        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing)
        self.graph.scene.render(painter)
        painter.end()
        img.save(path)

    # Replay & Impact
    def _replay(self):
        if not self._replay_tokens:
            QMessageBox.information(self, "Replay", "No tokens to replay yet.")
            return
        self.final_out.clear()
        app = QApplication.instance()
        for ch in self._replay_tokens:
            self.final_out.insertPlainText(ch)
            app.processEvents()
            time.sleep(0.005)

    def _explain_impact(self):
        now = self.final_out.toPlainText()
        d = diff_text(self._last_final, now)
        prompt = f"Given this diff, summarize impact in 2–3 bullets without revealing chain-of-thought:\n\n{d}"
        try:
            ans = []
            for ch in self.llm.chat_stream_final("Summarizer. No chain-of-thought.", prompt, temperature=0.2, max_tokens=120):
                ans.append(ch)
            QMessageBox.information(self, "Impact of Edit", "".join(ans))
        except Exception:
            QMessageBox.information(
                self, "Impact of Edit", f"(fallback)\n{d[:2000]}")

    # Why heatmap (proxy)
    def _render_heatmap(self, task_text: str, scores: Dict[str, float]):
        words = re.findall(r"\S+|\s+", task_text)
        html = []
        for w in words:
            base = re.sub(r"\W+$", "", w)
            key = base.lower()
            s = scores.get(key, 0.0)
            if s > 0:
                alpha = int(40 + 160 * s)
                html.append(
                    f"<span style='background-color: rgba(64,132,247,{alpha/255:.2f}); padding:1px 2px; border-radius:4px'>{w}</span>")
            else:
                html.append(w)
        self.why_out.setHtml("<b>Why Heatmap (proxy):</b><br>" + "".join(html))

    # serialization
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "alias": self.cfg.alias, "model_path": self.cfg.model_path, "n_ctx": self.cfg.n_ctx,
                "n_threads": self.cfg.n_threads, "n_gpu_layers": self.cfg.n_gpu_layers, "use_mock": self.cfg.use_mock,
                "use_ollama": self.cfg.use_ollama, "ollama_model": self.cfg.ollama_model,
            },
            "trace": self.trace.to_dict(),
            "final": self.final_out.toPlainText(),
        }

    def from_dict(self, data: Dict[str, Any]):
        cfg = data.get("config", {})
        self.alias_edit.setText(cfg.get("alias", self.cfg.alias))
        self.model_path_edit.setText(cfg.get("model_path", ""))
        self.ctx_spin.setValue(int(cfg.get("n_ctx", self.cfg.n_ctx)))
        self.th_spin.setValue(int(cfg.get("n_threads", self.cfg.n_threads)))
        self.gpu_spin.setValue(
            int(cfg.get("n_gpu_layers", self.cfg.n_gpu_layers)))
        self.mock_chk.setChecked(bool(cfg.get("use_mock", False)))
        self.ollama_chk.setChecked(bool(cfg.get("use_ollama", False)))
        self.ollama_model_edit.setText(cfg.get("ollama_model", ""))
        self.trace = TraceGraph.from_dict(data.get("trace", {}))
        self.final_out.setPlainText(data.get("final", ""))
        self._refresh_lists_and_graph()
        self.graph.zoom_fit()

# ============================= MAIN WINDOW ============================= #


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cognitive Sandbox — Local Visual Tracer")
        self.resize(1500, 950)
        self.session_log: List[Dict[str, Any]] = []

        tb = QToolBar()
        self.addToolBar(tb)
        act_save = QAction("Save Session", self)
        act_open = QAction("Load Session", self)
        act_prompts = QAction("Edit Prompts", self)
        act_bench = QAction("Run A/B Bench", self)
        act_about = QAction("About", self)
        tb.addAction(act_save)
        tb.addAction(act_open)
        tb.addSeparator()
        tb.addAction(act_prompts)
        tb.addAction(act_bench)
        tb.addSeparator()
        tb.addAction(act_about)
        act_save.triggered.connect(self.save_session)
        act_open.triggered.connect(self.load_session)
        act_prompts.triggered.connect(self.edit_prompts)
        act_bench.triggered.connect(self.run_bench)
        act_about.triggered.connect(self.show_about)

        # Query row
        top = QWidget()
        top_lay = QHBoxLayout(top)
        top_lay.addWidget(QLabel("Query / Task:"))
        self.query_edit = QTextEdit()
        self.query_edit.setPlaceholderText(
            "Ask anything… e.g., 'What is the capital of France and why is it significant?' ")
        self.query_edit.setFixedHeight(80)
        top_lay.addWidget(self.query_edit, 1)
        self.run_btn = QPushButton("Run Both ▶")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.run_both)
        top_lay.addWidget(self.run_btn)

        # Panels (pass safe getter instead of parent().parent())
        self.left_panel = ModelPanel(
            alias_default="gpt-oss-120b", get_task=lambda: self.query_edit.toPlainText().strip())
        self.right_panel = ModelPanel(
            alias_default="gpt-oss-20b", get_task=lambda: self.query_edit.toPlainText().strip())
        self.left_panel.run_complete.connect(
            lambda data: self._log_run("Left", data))
        self.right_panel.run_complete.connect(
            lambda data: self._log_run("Right", data))

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.left_panel)
        split.addWidget(self.right_panel)
        split.setSizes([750, 750])

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addWidget(top)
        lay.addWidget(split, 1)
        self.setCentralWidget(central)

    def _log_run(self, side: str, data: Dict[str, Any]):
        mem_mb = None
        if psutil:
            try:
                p = psutil.Process(os.getpid())
                mem_mb = int(p.memory_info().rss / (1024*1024))
            except Exception:
                pass
        entry = {"side": side, "alias": data.get("alias"), "timestamp": data.get("timestamp"), "task": data.get("task"),
                 "steps": data.get("steps"), "final": data.get("final"), "memory_rss_mb": mem_mb}
        self.session_log.append(entry)

    def run_both(self):
        task = self.query_edit.toPlainText().strip()
        if not task:
            QMessageBox.information(
                self, "Enter a query", "Please enter a task/question to run.")
            return
        self.left_panel.run_full(task)
        self.right_panel.run_full(task)

    def edit_prompts(self):
        PromptEditor(self).exec()

    # Bench
    def run_bench(self):
        prompts = MINI_BENCH[:5]
        panels = [("Left", self.left_panel), ("Right", self.right_panel)]
        results = []
        for label, panel in panels:
            if not (panel.cfg.use_mock or panel.cfg.use_ollama or (panel.cfg.model_path and panel.llm.llm is not None)):
                QMessageBox.information(
                    self, "Bench", f"{label} panel not loaded.")
                return
        for label, panel in panels:
            ok = 0
            total_t = 0.0
            total_tok = 0
            finals = []
            for p in prompts:
                worker = TraceWorker(panel.llm, p, None, None, prompts=PROMPTS)
                elapsed = {"t": 0.0, "tokens": 0, "final": ""}
                worker.metrics.connect(lambda m, e=elapsed: (e.update({"t": m.get(
                    "elapsed_sec", 0.0), "tokens": m.get("approx_tokens", 0), "final": m.get("final", "")}), None))
                worker.start()
                worker.wait()
                total_t += elapsed["t"]
                total_tok += elapsed["tokens"]
                f = elapsed["final"].strip()
                finals.append(f)
                if f:
                    ok += 1
            tps = round((total_tok/total_t), 2) if total_t > 0 else 0.0
            mem_mb = None
            if psutil:
                try:
                    p = psutil.Process(os.getpid())
                    mem_mb = int(p.memory_info().rss / (1024*1024))
                except Exception:
                    pass
            results.append((label, ok, round(total_t, 2),
                           total_tok, tps, mem_mb, finals))

        msg = "A/B Bench (5 prompts)\n\n"
        for label, ok, t, tok, tps, mem, _ in results:
            msg += f"{label}: accuracy~{ok}/5, total {t}s, tokens {tok}, ~{tps} tok/s"
            if mem is not None:
                msg += f", RSS≈{mem}MB"
            msg += "\n"

        data = [{"panel": label, "ok": ok, "total_sec": t, "tokens": tok, "tok_per_sec": tps,
                 "memory_rss_mb": mem, "finals": finals} for (label, ok, t, tok, tps, mem, finals) in results]
        try:
            with open("bench_results.json", "w", encoding="utf-8") as f:
                json.dump({"prompts": prompts, "results": data}, f, indent=2)
            msg += "\nSaved: bench_results.json"
        except Exception:
            pass
        QMessageBox.information(self, "Bench Results", msg)

    # Session I/O
    def save_session(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", filter="JSON (*.json)")
        if not path:
            return
        data = {"meta": {"app": "Cognitive Sandbox", "version": 2},
                "query": self.query_edit.toPlainText(),
                "left": self.left_panel.to_dict(),
                "right": self.right_panel.to_dict(),
                "log": self.session_log,
                "prompts": PROMPTS}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def load_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", filter="JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.query_edit.setPlainText(data.get("query", ""))
            self.left_panel.from_dict(data.get("left", {}))
            self.right_panel.from_dict(data.get("right", {}))
            self.session_log = data.get("log", [])
            # restore prompts if present
            pr = data.get("prompts")
            if isinstance(pr, dict):
                PROMPTS.update(pr)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def show_about(self):
        QMessageBox.information(self, "About Cognitive Sandbox",
                                """
Cognitive Sandbox — Local Visual LLM Tracer

• Compare two local models side-by-side
• Visual plan + editable notes (no CoT)
• Edit & re-run downstream
• Guardrail slider + optional grounding
• A/B micro-bench with JSON export
• Replay streamed answer
• Zoom-to-fit graph; HiDPI-safe
• Save/Load full sessions

Built with PySide6 + llama.cpp / Ollama.
            """)

# ============================= ENTRY ============================= #


def main():
    # HiDPI friendliness across platforms
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:
        pass
    app = QApplication(sys.argv)
    apply_theme(app, dark=True)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

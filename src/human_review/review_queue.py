"""
src/human_review/review_queue.py

Human-in-the-loop review queue for uncertain anomaly predictions.

When a model's prediction falls within the conformal prediction uncertainty
band (or MC-Dropout uncertainty exceeds a threshold), the image is added
to a persistent review queue.

A lightweight Gradio-based UI allows operators to:
  1. View the flagged image + anomaly heatmap
  2. Accept (label as normal) or Reject (label as defect)
  3. Export reviewed labels for active learning / fine-tuning
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Review item dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReviewItem:
    """Represents a single sample queued for human review."""
    item_id: str
    image_path: str
    anomaly_score: float
    uncertainty: float
    decision: str            # "review" (awaiting review)
    heatmap_path: str = ""   # path to saved heatmap PNG
    timestamp: float = field(default_factory=time.time)
    reviewer_label: Optional[str] = None  # "normal" or "defect" after review
    reviewer_notes: str = ""


# ---------------------------------------------------------------------------
# Review Queue
# ---------------------------------------------------------------------------

class ReviewQueue:
    """
    Persistent review queue backed by a JSON file.

    Stores uncertain predictions and provides methods to:
      - Add items to the queue
      - Mark items as reviewed
      - Export reviewed labels for model improvement

    Args:
        queue_path (str): Path to the JSON queue file.
    """

    def __init__(self, queue_path: str = "results/review_queue.json"):
        self.queue_path = Path(queue_path)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self._items: Dict[str, ReviewItem] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.queue_path.exists():
            with open(self.queue_path) as f:
                raw = json.load(f)
            self._items = {k: ReviewItem(**v) for k, v in raw.items()}

    def _save(self) -> None:
        with open(self.queue_path, "w") as f:
            json.dump({k: asdict(v) for k, v in self._items.items()}, f, indent=2)

    # ------------------------------------------------------------------
    def add(
        self,
        image_path: str,
        anomaly_score: float,
        uncertainty: float,
        heatmap_path: str = "",
    ) -> str:
        """
        Add an item to the review queue.

        Returns:
            item_id: Unique identifier for tracking.
        """
        item_id = f"{Path(image_path).stem}_{int(time.time() * 1000)}"
        item = ReviewItem(
            item_id=item_id,
            image_path=image_path,
            anomaly_score=anomaly_score,
            uncertainty=uncertainty,
            decision="review",
            heatmap_path=heatmap_path,
        )
        self._items[item_id] = item
        self._save()
        return item_id

    # ------------------------------------------------------------------
    def review(
        self,
        item_id: str,
        label: str,
        notes: str = "",
    ) -> None:
        """
        Mark an item as reviewed.

        Args:
            item_id: Item identifier.
            label: "normal" or "defect".
            notes: Optional reviewer notes.
        """
        assert label in ("normal", "defect"), "Label must be 'normal' or 'defect'"
        if item_id not in self._items:
            raise KeyError(f"Item {item_id} not found in queue")
        self._items[item_id].reviewer_label = label
        self._items[item_id].reviewer_notes = notes
        self._save()

    # ------------------------------------------------------------------
    def pending(self) -> List[ReviewItem]:
        """Return all items awaiting review."""
        return [item for item in self._items.values() if item.reviewer_label is None]

    def reviewed(self) -> List[ReviewItem]:
        """Return all reviewed items."""
        return [item for item in self._items.values() if item.reviewer_label is not None]

    # ------------------------------------------------------------------
    def export_labels(self, output_path: str = "results/reviewed_labels.json") -> Path:
        """
        Export reviewed labels for active learning.

        Returns a JSON manifest with image paths and verified labels.
        """
        labels = [
            {
                "image_path": item.image_path,
                "label": 1 if item.reviewer_label == "defect" else 0,
                "anomaly_score": item.anomaly_score,
                "uncertainty": item.uncertainty,
                "notes": item.reviewer_notes,
            }
            for item in self.reviewed()
        ]
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(labels, f, indent=2)
        return out

    # ------------------------------------------------------------------
    def summary(self) -> Dict:
        total = len(self._items)
        pending = len(self.pending())
        reviewed = len(self.reviewed())
        defects = sum(1 for i in self.reviewed() if i.reviewer_label == "defect")
        return {
            "total": total,
            "pending": pending,
            "reviewed": reviewed,
            "confirmed_defects": defects,
            "confirmed_normal": reviewed - defects,
        }


# ---------------------------------------------------------------------------
# Gradio Review UI
# ---------------------------------------------------------------------------

def launch_review_ui(queue: ReviewQueue, port: int = 7860) -> None:
    """
    Launch a Gradio-based human review interface.

    Operators see the image + heatmap and can label it as normal or defect.

    Args:
        queue: The ReviewQueue instance.
        port: Web server port.
    """
    try:
        import gradio as gr
        from PIL import Image
    except ImportError:
        print("Install gradio and Pillow to use the review UI: pip install gradio Pillow")
        return

    pending_items = queue.pending()
    if not pending_items:
        print("No items pending review.")
        return

    state = {"index": 0, "items": pending_items}

    def load_item(index: int):
        if index >= len(state["items"]):
            return None, None, "Queue empty", 0.0, 0.0
        item = state["items"][index]
        img = Image.open(item.image_path).convert("RGB") if Path(item.image_path).exists() else None
        hmap = Image.open(item.heatmap_path).convert("RGB") if (
            item.heatmap_path and Path(item.heatmap_path).exists()
        ) else None
        return img, hmap, item.item_id, item.anomaly_score, item.uncertainty

    def submit_review(item_id: str, label: str, notes: str):
        if item_id:
            queue.review(item_id, label.lower(), notes)
            state["index"] += 1
        return load_item(state["index"])

    with gr.Blocks(title="VisionInspect AI — Human Review") as demo:
        gr.Markdown("# 🔍 VisionInspect AI — Human Review Queue")

        with gr.Row():
            img_display = gr.Image(label="Product Image")
            hmap_display = gr.Image(label="Anomaly Heatmap")

        with gr.Row():
            item_id_box = gr.Textbox(label="Item ID", interactive=False)
            score_box = gr.Number(label="Anomaly Score", interactive=False)
            unc_box = gr.Number(label="Uncertainty", interactive=False)

        label_input = gr.Radio(["Normal", "Defect"], label="Your Label")
        notes_input = gr.Textbox(label="Notes (optional)")
        submit_btn = gr.Button("Submit Review →")
        status = gr.Markdown("")

        demo.load(lambda: load_item(0), outputs=[img_display, hmap_display, item_id_box, score_box, unc_box])
        submit_btn.click(
            submit_review,
            inputs=[item_id_box, label_input, notes_input],
            outputs=[img_display, hmap_display, item_id_box, score_box, unc_box],
        )

    demo.launch(server_port=port, share=False)

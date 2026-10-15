"""
app.py — VisionInspect AI Web Dashboard  (frontend v2)

Interactive web interface for industrial defect detection:
  1. Live Inspection   – heatmap / overlay / boxes, score gauge, defect-region table,
                         adjustable sensitivity, annotated-image download, session history
  2. Batch Inspection  – drop many images, get a gallery, summary and CSV export
  3. Review Queue      – triage uncertain items (skip / confirm), pending + history tables, export
  4. Benchmarks        – metric cards, per-run selector and cross-run comparison table
  5. Guide             – how the verdict logic works

NOTE: Only this frontend file was changed. The backend (src/, train.py, evaluate.py ...)
is used exactly as before — same imports, same calls (build_model, PatchCore/PaDiM.load,
model.predict, AnomalyMapProcessor, ReviewQueue).

Launch:
    python app.py
"""

import sys
import csv
import json
import time
import html
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import torch
from PIL import Image
import gradio as gr

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.transforms import get_val_transforms, denormalize  # noqa: F401  (kept from original)
from models import build_model
from localization.anomaly_map import AnomalyMapProcessor
from human_review.review_queue import ReviewQueue, ReviewItem  # noqa: F401
from uncertainty.conformal import ConformalPredictor  # noqa: F401

# ---------------------------------------------------------------------------
# Backend objects (unchanged behaviour)
# ---------------------------------------------------------------------------
queue = ReviewQueue(queue_path="results/review_queue.json")
processor = AnomalyMapProcessor(sigma=4.0, normalize=True)
transform = get_val_transforms()

MODEL_CACHE: Dict[str, Any] = {}

MODELS = ["autoencoder", "padim", "patchcore"]
CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
              "metal_nut", "pill", "screw", "tile", "wood", "zipper"]
DEFAULT_THRESHOLD = 2.59
DEFAULT_T_NORMAL = 2.5849
DEFAULT_T_ANOMALY = 2.6085

EXPORT_DIR = Path("results/exports")


def get_cached_model(model_name: str, category: str, device: str = "cpu"):
    key = f"{model_name}_{category}_{device}"
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    model_path = Path("results/models") / model_name / category / "model.pt"
    if not model_path.exists():
        return None

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    if model_name == "patchcore":
        from models.patchcore import PatchCore
        model = PatchCore.load(str(model_path), device=str(dev))
    elif model_name == "padim":
        from models.padim import PaDiM
        model = PaDiM.load(str(model_path), device=str(dev))
    else:
        model = build_model(model_name, device=str(dev))
        state = torch.load(str(model_path), map_location=dev)
        model.load_state_dict(state)
        model.to(dev).eval()

    MODEL_CACHE[key] = (model, dev)
    return MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
:root{
  --vi-ok:#16a34a; --vi-ok-bg:#dcfce7; --vi-ok-fg:#14532d;
  --vi-bad:#dc2626; --vi-bad-bg:#fee2e2; --vi-bad-fg:#7f1d1d;
  --vi-warn:#d97706; --vi-warn-bg:#fef3c7; --vi-warn-fg:#78350f;
  --vi-idle-bg:#f1f5f9; --vi-idle-fg:#334155;
  --vi-card:#ffffff; --vi-border:#e2e8f0; --vi-muted:#64748b;
}
.dark{ --vi-ok-bg:#052e16; --vi-ok-fg:#bbf7d0; --vi-bad-bg:#450a0a; --vi-bad-fg:#fecaca;
  --vi-warn-bg:#451a03; --vi-warn-fg:#fde68a; --vi-idle-bg:#1e293b; --vi-idle-fg:#cbd5e1;
  --vi-card:#0f172a; --vi-border:#334155; --vi-muted:#94a3b8; }
.gradio-container{max-width:1400px !important;margin:auto;}
.vi-hero{background:linear-gradient(120deg,#0f172a 0%,#1e3a8a 55%,#2563eb 100%);
  color:#fff;padding:26px 30px;border-radius:16px;margin-bottom:8px;
  box-shadow:0 8px 24px rgba(30,58,138,.25);}
.vi-hero h1{margin:0;font-size:1.9em;font-weight:800;letter-spacing:-.5px;color:#fff;}
.vi-hero p{margin:6px 0 0;opacity:.85;font-size:1.0em;color:#fff;}
.vi-chips{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;}
.vi-chip{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);
  padding:3px 12px;border-radius:999px;font-size:.82em;color:#fff;}
.vi-banner{padding:18px;border-radius:12px;font-size:1.25em;font-weight:700;text-align:center;border:2px solid;}
.vi-ok{background:var(--vi-ok-bg);color:var(--vi-ok-fg);border-color:var(--vi-ok);}
.vi-bad{background:var(--vi-bad-bg);color:var(--vi-bad-fg);border-color:var(--vi-bad);}
.vi-warn{background:var(--vi-warn-bg);color:var(--vi-warn-fg);border-color:var(--vi-warn);}
.vi-idle{background:var(--vi-idle-bg);color:var(--vi-idle-fg);border-color:var(--vi-border);font-weight:500;font-size:1em;}
.vi-banner small{display:block;font-weight:400;font-size:.62em;margin-top:4px;opacity:.85;}
.vi-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:10px 0;}
.vi-card{background:var(--vi-card);border:1px solid var(--vi-border);border-radius:12px;padding:12px 14px;}
.vi-card .k{font-size:.74em;color:var(--vi-muted);text-transform:uppercase;letter-spacing:.06em;}
.vi-card .v{font-size:1.5em;font-weight:700;margin-top:2px;}
.vi-card .s{font-size:.78em;color:var(--vi-muted);}
.vi-gauge{margin:10px 0 4px;}
.vi-gauge .track{position:relative;height:16px;border-radius:999px;overflow:hidden;
  background:linear-gradient(90deg,#16a34a 0%,#16a34a var(--n),#f59e0b var(--n),#f59e0b var(--a),#dc2626 var(--a),#dc2626 100%);}
.vi-gauge .mark{position:absolute;top:-4px;width:4px;height:24px;background:#0f172a;border-radius:2px;
  box-shadow:0 0 0 2px #fff;left:var(--p);transform:translateX(-2px);}
.vi-gauge .lbl{display:flex;justify-content:space-between;font-size:.74em;color:var(--vi-muted);margin-top:6px;}
.vi-note{font-size:.88em;color:var(--vi-muted);}
.vi-footer{text-align:center;color:var(--vi-muted);font-size:.8em;margin-top:14px;}
"""


def _banner(kind: str, text: str, sub: str = "") -> str:
    sub_html = f"<small>{sub}</small>" if sub else ""
    return f"<div class='vi-banner vi-{kind}'>{text}{sub_html}</div>"


IDLE_BANNER = _banner("idle", "Upload an image (or click a sample) and press <b>Run Defect Inspection</b>.")


def _cards(items: List[tuple]) -> str:
    out = "<div class='vi-cards'>"
    for k, v, s in items:
        out += f"<div class='vi-card'><div class='k'>{html.escape(str(k))}</div><div class='v'>{v}</div><div class='s'>{s}</div></div>"
    return out + "</div>"


def _gauge(score: float, t_normal: float, t_anom: float, conformal: bool) -> str:
    """Score bar showing normal / uncertain / defect zones and where this sample falls."""
    hi = max(score, t_anom, t_normal) * 1.25 + 1e-6
    pct = lambda x: max(0.0, min(100.0, 100.0 * x / hi))
    n, a = (pct(t_normal), pct(t_anom)) if conformal else (pct(t_normal), pct(t_normal))
    return (
        f"<div class='vi-gauge'><div class='track' style='--n:{n:.1f}%;--a:{a:.1f}%;'>"
        f"<div class='mark' style='--p:{pct(score):.1f}%;'></div></div>"
        f"<div class='lbl'><span>0 &nbsp;(normal)</span><span>score {score:.3f}</span><span>{hi:.2f} (defect)</span></div></div>"
    )


# ---------------------------------------------------------------------------
# Helpers that read backend output files (read-only)
# ---------------------------------------------------------------------------

def _metrics_path(model: str, category: str) -> Path:
    return Path("results/eval") / model / category / "metrics.json"


def _read_metrics(model: str, category: str) -> Optional[dict]:
    p = _metrics_path(model, category)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def available_models() -> List[str]:
    """Models/categories whose weights exist — purely informational."""
    return [m for m in MODELS if any((Path("results/models") / m / c / "model.pt").exists() for c in CATEGORIES)]


def auto_threshold(model: str, category: str):
    """Pull the optimal threshold saved by evaluate.py, if available."""
    m = _read_metrics(model, category)
    if m and "optimal_threshold" in m:
        t = float(m["optimal_threshold"])
        return gr.update(value=round(t, 4)), f"✅ Threshold loaded from evaluation: **{t:.4f}** (optimal F1)"
    return gr.update(), "ℹ️ No evaluation metrics for this model/category — keeping current threshold."


def model_status(model: str, category: str) -> str:
    ok = (Path("results/models") / model / category / "model.pt").exists()
    return ("🟢 Model weights found." if ok else
            f"🔴 No weights at `results/models/{model}/{category}/model.pt` — train first (`python train.py`).")


def decide(score: float, threshold: float, conformal: bool, t_n: float, t_a: float) -> str:
    if conformal:
        if score < t_n:
            return "NORMAL"
        if score >= t_a:
            return "DEFECT"
        return "REVIEW"
    return "DEFECT" if score >= threshold else "NORMAL"


def verdict_banner(verdict: str, n_regions: int) -> str:
    if verdict == "NORMAL":
        return _banner("ok", "✅ ACCEPTED — NORMAL PRODUCT")
    if verdict == "DEFECT":
        return _banner("bad", f"❌ REJECTED — DEFECT DETECTED",
                       f"{n_regions} suspicious region{'s' if n_regions != 1 else ''} localised")
    return _banner("warn", "⚠️ UNCERTAIN — FLAGGED FOR HUMAN OPERATOR REVIEW",
                   "Score lies inside the conformal uncertainty band")


# ---------------------------------------------------------------------------
# Core inference (shared by single + batch)
# ---------------------------------------------------------------------------

def run_inference(img: Image.Image, model_name, category, threshold, device,
                  conformal, t_n, t_a, box_sens, min_area, overlay_alpha) -> Optional[Dict[str, Any]]:
    model_info = get_cached_model(model_name, category, device)
    if model_info is None:
        return None
    model, dev = model_info

    img_rgb = img.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0).to(dev)

    t0 = time.perf_counter()
    with torch.no_grad():
        amap_raw, score_tensor = model.predict(tensor)
    latency_ms = (time.perf_counter() - t0) * 1000

    score = float(score_tensor[0].cpu())
    w, h = img.size

    amap_np = amap_raw[0, 0].cpu().numpy()
    amap_np = cv2.resize(amap_np, (w, h))
    amap_norm = processor.process(amap_np)

    heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    orig_np = np.array(img_rgb)
    a = float(overlay_alpha)
    overlay = ((1 - a) * orig_np + a * heatmap_rgb).clip(0, 255).astype(np.uint8)

    binary_mask, bboxes = processor.threshold_map(amap_norm, threshold=float(box_sens), min_area=int(min_area))
    verdict = decide(score, threshold, conformal, t_n, t_a)
    box_color = (255, 30, 30) if verdict != "NORMAL" else (40, 200, 90)
    bbox_img = processor.draw_bboxes(orig_np, bboxes, color=box_color, thickness=max(2, w // 200))

    regions = []
    for i, (x, y, bw, bh) in enumerate(bboxes, 1):
        reg = amap_norm[y:y + bh, x:x + bw]
        regions.append({
            "id": i, "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
            "area_px": int(bw * bh),
            "area_pct": round(100.0 * bw * bh / (w * h), 2),
            "peak": round(float(reg.max()), 3),
        })

    return dict(score=score, latency=latency_ms, verdict=verdict, regions=regions,
                heatmap=heatmap_rgb, overlay=overlay, bbox=bbox_img, size=(w, h))


# ---------------------------------------------------------------------------
# Tab 1: Live inspection
# ---------------------------------------------------------------------------

HISTORY_COLS = ["Time", "Model", "Category", "Score", "Verdict", "Regions", "Latency (ms)"]


def _history_rows(hist: list):
    return [[r[c] for c in HISTORY_COLS] for r in hist[::-1]]


def inspect_product(input_image, model_name, category, threshold, device,
                    enable_conformal, t_normal, t_anomaly,
                    box_sens, min_area, overlay_alpha, auto_queue, history, last):
    empty = (None, None, None, None)
    hist = list(history or [])

    if input_image is None:
        return (_banner("bad", "⚠️ Please upload an image or select a sample."), "", "",
                *empty, [], _history_rows(hist), hist, last, "")

    res = run_inference(input_image, model_name, category, threshold, device,
                        enable_conformal, t_normal, t_anomaly, box_sens, min_area, overlay_alpha)
    if res is None:
        return (_banner("warn", "⚠️ Model weights not found",
                        f"Expected <code>results/models/{model_name}/{category}/model.pt</code> — train the model first."),
                "", "", *empty, [], _history_rows(hist), hist, last, "")

    verdict, score = res["verdict"], res["score"]
    cards = _cards([
        ("Anomaly score", f"{score:.4f}", f"threshold {threshold:.4f}"),
        ("Verdict", verdict, "conformal routing" if enable_conformal else "single threshold"),
        ("Defect regions", len(res["regions"]), f"sensitivity {box_sens:.2f}"),
        ("Latency", f"{res['latency']:.1f} ms", f"{model_name} · {category}"),
        ("Image size", f"{res['size'][0]}×{res['size'][1]}", "pixels"),
    ])
    gauge = _gauge(score, t_normal if enable_conformal else threshold, t_anomaly, enable_conformal)
    gauge_html = gauge + (
        "<div class='vi-note'>🟩 certain normal · 🟧 uncertain (human review) · 🟥 certain defect</div>"
        if enable_conformal else "<div class='vi-note'>🟩 below threshold · 🟥 above threshold</div>")

    # table rows
    rows = [[r["id"], r["x"], r["y"], r["w"], r["h"], r["area_px"], r["area_pct"], r["peak"]] for r in res["regions"]]

    # history entry
    hist.append({"Time": datetime.now().strftime("%H:%M:%S"), "Model": model_name, "Category": category,
                 "Score": round(score, 4), "Verdict": verdict, "Regions": len(res["regions"]),
                 "Latency (ms)": round(res["latency"], 1)})

    # optional auto-queue of uncertain samples
    queue_msg = ""
    if auto_queue and verdict == "REVIEW":
        queue_msg = send_to_review_queue(input_image, {"Anomaly Score": score})

    last_state = {"image": input_image, "score": score, "verdict": verdict, "bbox": res["bbox"],
                  "overlay": res["overlay"], "heatmap": res["heatmap"], "model": model_name,
                  "category": category, "threshold": threshold, "regions": res["regions"],
                  "latency": res["latency"]}

    return (verdict_banner(verdict, len(res["regions"])), cards, gauge_html,
            input_image, res["heatmap"], res["overlay"], res["bbox"],
            rows, _history_rows(hist), hist, last_state, queue_msg)


def send_to_review_queue(input_image, last):
    """`last` is either the inspection state dict or {'Anomaly Score': x}."""
    if input_image is None or not last:
        return "⚠️ No active inspection to queue."
    score = last.get("score", last.get("Anomaly Score"))
    if score is None:
        return "⚠️ No active inspection to queue."

    temp_dir = Path("results/inspection")
    temp_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    img_path = temp_dir / f"flagged_{ts}.png"
    input_image.save(img_path)

    item_id = queue.add(image_path=str(img_path), anomaly_score=float(score),
                        uncertainty=0.05, heatmap_path="")
    return f"✅ Sample queued successfully! Item ID: `{item_id}`"


def save_report(last):
    """Write annotated image + JSON report for download."""
    if not last:
        return None
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = EXPORT_DIR / f"inspection_{ts}.png"
    Image.fromarray(last["bbox"]).save(img_path)
    with open(EXPORT_DIR / f"inspection_{ts}.json", "w") as f:
        json.dump({k: last[k] for k in ("model", "category", "score", "threshold", "verdict", "latency", "regions")},
                  f, indent=2)
    return str(img_path)


def export_history(hist):
    if not hist:
        return None
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    p = EXPORT_DIR / f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=HISTORY_COLS)
        wr.writeheader()
        wr.writerows(hist)
    return str(p)


def clear_history():
    return [], [], None


# ---------------------------------------------------------------------------
# Tab 2: Batch inspection
# ---------------------------------------------------------------------------

BATCH_COLS = ["File", "Score", "Verdict", "Regions", "Latency (ms)"]


def batch_inspect(files, model_name, category, threshold, device, conformal, t_n, t_a,
                  box_sens, min_area, progress=gr.Progress()):
    if not files:
        return "<div class='vi-note'>Upload one or more images first.</div>", [], [], None

    if get_cached_model(model_name, category, device) is None:
        return _banner("warn", "⚠️ Model weights not found", "Train the model first."), [], [], None

    rows, gallery, counts = [], [], {"NORMAL": 0, "DEFECT": 0, "REVIEW": 0}
    paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]
    for p in progress.tqdm(paths, desc="Inspecting"):
        try:
            img = Image.open(p)
            res = run_inference(img, model_name, category, threshold, device,
                                conformal, t_n, t_a, box_sens, min_area, 0.4)
        except Exception as e:  # unreadable file
            rows.append([Path(p).name, "—", f"ERROR: {e}", 0, 0])
            continue
        counts[res["verdict"]] += 1
        rows.append([Path(p).name, round(res["score"], 4), res["verdict"],
                     len(res["regions"]), round(res["latency"], 1)])
        gallery.append((res["bbox"], f"{Path(p).name} · {res['verdict']} · {res['score']:.3f}"))

    total = sum(counts.values())
    rej = 100.0 * counts["DEFECT"] / total if total else 0.0
    summary = _cards([
        ("Inspected", total, "images"),
        ("Accepted", counts["NORMAL"], "normal"),
        ("Rejected", counts["DEFECT"], f"{rej:.1f}% reject rate"),
        ("Review", counts["REVIEW"], "uncertain"),
    ])

    csv_path = None
    if rows:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = str(EXPORT_DIR / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(BATCH_COLS)
            wr.writerows(rows)
    return summary, rows, gallery, csv_path


# ---------------------------------------------------------------------------
# Tab 3: Review queue
# ---------------------------------------------------------------------------

PENDING_COLS = ["Item ID", "Score", "Queued at"]
REVIEWED_COLS = ["Item ID", "Score", "Label", "Notes"]


def _summary_md() -> str:
    s = queue.summary()
    return (f"📊 **{s['pending']} Pending** · **{s['reviewed']} Reviewed** "
            f"(Defects: {s['confirmed_defects']}, Normal: {s['confirmed_normal']}) · Total: {s['total']}")


def _tables():
    pend = [[i.item_id, round(i.anomaly_score, 4), datetime.fromtimestamp(i.timestamp).strftime("%Y-%m-%d %H:%M")]
            for i in queue.pending()]
    rev = [[i.item_id, round(i.anomaly_score, 4), i.reviewer_label, i.reviewer_notes] for i in queue.reviewed()]
    return pend, rev


def load_next_review_item(skip: int = 0):
    pending = queue.pending()
    pend, rev = _tables()
    summary_text = _summary_md()
    if not pending:
        return None, "", "", 0.0, summary_text, 0, pend, rev, "🎉 All caught up! No items pending review."
    idx = skip % len(pending)
    item = pending[idx]
    img = Image.open(item.image_path) if Path(item.image_path).exists() else None
    note = "" if img is not None else "⚠️ Image file missing on disk."
    return img, item.item_id, "", item.anomaly_score, summary_text, idx, pend, rev, note


def submit_human_decision(item_id, decision, notes):
    msg = ""
    if item_id:
        label = "defect" if decision == "Confirm Defect" else "normal"
        try:
            queue.review(item_id, label, notes)
            msg = f"Saved **{label}** for `{item_id}`."
        except Exception as e:
            msg = f"⚠️ Error reviewing {item_id}: {e}"
    out = list(load_next_review_item(0))
    if msg and not out[8]:
        out[8] = msg
    return tuple(out)


def skip_item(idx):
    return load_next_review_item(int(idx or 0) + 1)


def export_reviewed_data():
    out = queue.export_labels("results/reviewed_labels.json")
    return f"✅ Exported verified labels to `{out}`", str(out)


# ---------------------------------------------------------------------------
# Tab 4: Benchmarks
# ---------------------------------------------------------------------------

def list_eval_runs() -> List[str]:
    base = Path("results/eval")
    if not base.exists():
        return []
    return sorted(f"{p.parent.parent.name}/{p.parent.name}" for p in base.glob("*/*/metrics.json"))


METRIC_LABELS = {"image_auroc": "Image AUROC", "image_ap": "Image AP", "image_f1": "Image F1",
                 "pixel_auroc": "Pixel AUROC", "pro_score": "PRO score", "fpr_at_95tpr": "FPR @95 TPR",
                 "optimal_threshold": "Opt. threshold", "latency_ms": "Latency (ms)"}


def load_metrics_report(run: Optional[str] = None):
    runs = list_eval_runs()
    if not runs:
        return {"Status": "No evaluation metrics found. Run evaluate.py first."}, "<div class='vi-note'>No evaluation runs found.</div>"
    run = run if run in runs else runs[0]
    model, cat = run.split("/")
    m = _read_metrics(model, cat) or {}
    cards = _cards([(METRIC_LABELS.get(k, k), f"{v:.4f}" if isinstance(v, (int, float)) else str(v), run)
                    for k, v in m.items() if isinstance(v, (int, float))])
    return m, cards


def comparison_table():
    rows = []
    for run in list_eval_runs():
        model, cat = run.split("/")
        m = _read_metrics(model, cat) or {}
        g = lambda k: round(float(m[k]), 4) if k in m and isinstance(m[k], (int, float)) else None
        rows.append([model, cat, g("image_auroc"), g("image_ap"), g("image_f1"),
                     g("pixel_auroc"), g("latency_ms")])
    return rows


def refresh_benchmarks(run):
    runs = list_eval_runs()
    m, cards = load_metrics_report(run)
    summ = None
    if runs:
        r = run if run in runs else runs[0]
        model, cat = r.split("/")
        p = Path("results/eval") / model / cat / "visualizations" / "summary.png"
        summ = str(p) if p.exists() else None
    return gr.update(choices=runs, value=(run if run in runs else (runs[0] if runs else None))), m, cards, comparison_table(), summ


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_app():
    sample_images = []
    base_mvtec = Path("data/raw/mvtec")
    if base_mvtec.exists():
        for p in sorted(base_mvtec.glob("*/test/*/*.png"))[:12]:
            sample_images.append(str(p))

    with gr.Blocks(title="VisionInspect AI — Industrial Inspection Dashboard") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        history_state = gr.State([])
        last_state = gr.State(None)

        gr.HTML(
            "<div class='vi-hero'><h1>🔍 VisionInspect AI</h1>"
            "<p>Industrial quality inspection — anomaly detection, defect localisation &amp; uncertainty-guided human review</p>"
            "<div class='vi-chips'><span class='vi-chip'>Autoencoder · PaDiM · PatchCore</span>"
            "<span class='vi-chip'>Conformal routing</span><span class='vi-chip'>Human-in-the-loop</span></div></div>"
        )

        with gr.Tabs():
            # ================= TAB 1: LIVE =================
            with gr.Tab("🔬 Live Inspection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(type="pil", label="Product image", sources=["upload", "clipboard", "webcam"])
                        with gr.Row():
                            model_choice = gr.Dropdown(MODELS, value="autoencoder", label="Model")
                            category_choice = gr.Dropdown(CATEGORIES, value="bottle", label="Category")
                        model_status_md = gr.Markdown(model_status("autoencoder", "bottle"))
                        thresh_slider = gr.Slider(0.0, 5.0, value=DEFAULT_THRESHOLD, step=0.01,
                                                  label="Anomaly score threshold")
                        auto_thr_md = gr.Markdown("")
                        auto_thr_btn = gr.Button("🎯 Use optimal threshold from evaluation", size="sm")
                        device_choice = gr.Radio(["cpu", "cuda"], value="cpu", label="Compute device")

                        with gr.Accordion("🎯 Conformal prediction (uncertainty routing)", open=False):
                            conformal_check = gr.Checkbox(value=True, label="Enable uncertainty routing")
                            t_norm_slider = gr.Slider(0.0, 5.0, value=DEFAULT_T_NORMAL, step=0.0001,
                                                      label="T_Normal (certain normal below)")
                            t_anom_slider = gr.Slider(0.0, 5.0, value=DEFAULT_T_ANOMALY, step=0.0001,
                                                      label="T_Anomaly (certain defect above)")
                            auto_queue_check = gr.Checkbox(value=False,
                                                           label="Auto-send UNCERTAIN results to review queue")

                        with gr.Accordion("🖌️ Localisation & display", open=False):
                            box_slider = gr.Slider(0.1, 0.95, value=0.45, step=0.01,
                                                   label="Box sensitivity (lower = more regions)")
                            area_slider = gr.Slider(5, 2000, value=30, step=5, label="Minimum region area (px)")
                            alpha_slider = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="Heatmap opacity")

                        with gr.Row():
                            inspect_btn = gr.Button("⚡ Run Defect Inspection", variant="primary", size="lg", scale=3)
                            clear_btn = gr.Button("🧹 Clear", scale=1)

                        if sample_images:
                            gr.Examples(examples=sample_images, inputs=img_input, label="Samples (click to load)")

                    with gr.Column(scale=2):
                        verdict_html = gr.HTML(IDLE_BANNER)
                        cards_html = gr.HTML("")
                        gauge_html = gr.HTML("")

                        with gr.Tabs():
                            with gr.Tab("Evidence"):
                                with gr.Row():
                                    out_orig = gr.Image(label="Original", interactive=False)
                                    out_hmap = gr.Image(label="Anomaly heatmap", interactive=False)
                                with gr.Row():
                                    out_overlay = gr.Image(label="Overlay", interactive=False)
                                    out_bbox = gr.Image(label="Defect localisation", interactive=False)
                            with gr.Tab("Defect regions"):
                                regions_df = gr.Dataframe(
                                    headers=["#", "x", "y", "width", "height", "area px", "area %", "peak intensity"],
                                    interactive=False, wrap=True)
                            with gr.Tab("Session history"):
                                history_df = gr.Dataframe(headers=HISTORY_COLS, interactive=False, wrap=True)
                                with gr.Row():
                                    hist_export_btn = gr.Button("💾 Export history (CSV)", size="sm")
                                    hist_clear_btn = gr.Button("🗑️ Clear history", size="sm")
                                hist_file = gr.File(label="History CSV", interactive=False)

                        with gr.Row():
                            queue_btn = gr.Button("📤 Send to human review queue", variant="secondary")
                            report_btn = gr.Button("📥 Save annotated report", variant="secondary")
                        queue_status = gr.Markdown("")
                        report_file = gr.File(label="Annotated image", interactive=False)

                inspect_inputs = [img_input, model_choice, category_choice, thresh_slider, device_choice,
                                  conformal_check, t_norm_slider, t_anom_slider,
                                  box_slider, area_slider, alpha_slider, auto_queue_check, history_state, last_state]
                inspect_outputs = [verdict_html, cards_html, gauge_html, out_orig, out_hmap, out_overlay, out_bbox,
                                   regions_df, history_df, history_state, last_state, queue_status]
                inspect_btn.click(inspect_product, inputs=inspect_inputs, outputs=inspect_outputs)

                # Re-run when the sample is picked
                img_input.upload(inspect_product, inputs=inspect_inputs, outputs=inspect_outputs)

                for dd in (model_choice, category_choice):
                    dd.change(model_status, [model_choice, category_choice], model_status_md)
                auto_thr_btn.click(auto_threshold, [model_choice, category_choice], [thresh_slider, auto_thr_md])

                def _clear():
                    return (None, IDLE_BANNER, "", "", None, None, None, None, [], None, "")
                clear_btn.click(_clear, outputs=[img_input, verdict_html, cards_html, gauge_html, out_orig,
                                                 out_hmap, out_overlay, out_bbox, regions_df, last_state, queue_status])

                queue_btn.click(send_to_review_queue, [img_input, last_state], [queue_status])
                report_btn.click(save_report, [last_state], [report_file])
                hist_export_btn.click(export_history, [history_state], [hist_file])
                hist_clear_btn.click(clear_history, outputs=[history_state, history_df, hist_file])

            # ================= TAB 2: BATCH =================
            with gr.Tab("🗂️ Batch Inspection"):
                gr.Markdown("Inspect many images at once using the model, category and thresholds chosen on the "
                            "**Live Inspection** tab.")
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_files = gr.File(file_count="multiple", file_types=["image"], label="Drop images here",
                                              type="filepath")
                        batch_btn = gr.Button("🚀 Run batch inspection", variant="primary", size="lg")
                        batch_csv = gr.File(label="Results CSV", interactive=False)
                    with gr.Column(scale=2):
                        batch_summary = gr.HTML("")
                        batch_df = gr.Dataframe(headers=BATCH_COLS, interactive=False, wrap=True)
                        batch_gallery = gr.Gallery(label="Localisation results", columns=4, height="auto")
                batch_btn.click(batch_inspect,
                                [batch_files, model_choice, category_choice, thresh_slider, device_choice,
                                 conformal_check, t_norm_slider, t_anom_slider, box_slider, area_slider],
                                [batch_summary, batch_df, batch_gallery, batch_csv])

            # ================= TAB 3: REVIEW =================
            with gr.Tab("👨‍💼 Human Review Queue"):
                gr.Markdown("Review borderline predictions and uncertain edge cases to avoid false rejections.")
                queue_summary_box = gr.Markdown(_summary_md())
                skip_idx = gr.State(0)

                with gr.Row():
                    with gr.Column(scale=1):
                        review_img = gr.Image(label="Flagged product", interactive=False)
                        item_id_box = gr.Textbox(label="Queue item ID", interactive=False)
                        item_score_box = gr.Number(label="Model anomaly score", interactive=False)
                    with gr.Column(scale=1):
                        gr.Markdown("#### Operator verdict")
                        notes_box = gr.Textbox(lines=3, label="Inspection notes",
                                               placeholder="e.g. Minor reflection, surface scratch confirmed")
                        with gr.Row():
                            btn_normal = gr.Button("✅ Confirm Normal (false alarm)", variant="primary")
                            btn_defect = gr.Button("❌ Confirm Defect (reject)", variant="stop")
                        with gr.Row():
                            skip_btn = gr.Button("⏭️ Skip")
                            refresh_btn = gr.Button("🔄 Refresh")
                            export_btn = gr.Button("💾 Export reviewed (JSON)")
                        review_msg = gr.Markdown("")
                        export_file = gr.File(label="Exported labels", interactive=False)

                with gr.Tabs():
                    with gr.Tab("Pending"):
                        pending_df = gr.Dataframe(headers=PENDING_COLS, interactive=False, wrap=True)
                    with gr.Tab("Reviewed"):
                        reviewed_df = gr.Dataframe(headers=REVIEWED_COLS, interactive=False, wrap=True)

                review_outputs = [review_img, item_id_box, notes_box, item_score_box, queue_summary_box,
                                  skip_idx, pending_df, reviewed_df, review_msg]
                btn_normal.click(lambda i, n: submit_human_decision(i, "Confirm Normal", n),
                                 [item_id_box, notes_box], review_outputs)
                btn_defect.click(lambda i, n: submit_human_decision(i, "Confirm Defect", n),
                                 [item_id_box, notes_box], review_outputs)
                skip_btn.click(skip_item, [skip_idx], review_outputs)
                refresh_btn.click(lambda: load_next_review_item(0), outputs=review_outputs)
                export_btn.click(export_reviewed_data, outputs=[review_msg, export_file])
                demo.load(lambda: load_next_review_item(0), outputs=review_outputs)

            # ================= TAB 4: BENCHMARKS =================
            with gr.Tab("📈 Benchmarks"):
                gr.Markdown("### Evaluation results (from `evaluate.py`)")
                runs0 = list_eval_runs()
                with gr.Row():
                    run_dd = gr.Dropdown(runs0, value=(runs0[0] if runs0 else None), label="Evaluation run (model/category)")
                    reload_btn = gr.Button("🔄 Reload", scale=0)
                m0, c0 = load_metrics_report(runs0[0] if runs0 else None)
                metric_cards = gr.HTML(c0)
                with gr.Row():
                    with gr.Column():
                        metrics_json_view = gr.JSON(value=m0, label="Raw metrics")
                    with gr.Column():
                        summary_img = gr.Image(label="Evaluation summary plot", interactive=False)
                gr.Markdown("#### All runs compared")
                compare_df = gr.Dataframe(
                    value=comparison_table(),
                    headers=["Model", "Category", "Image AUROC", "Image AP", "Image F1", "Pixel AUROC", "Latency (ms)"],
                    interactive=False, wrap=True)
                bench_out = [run_dd, metrics_json_view, metric_cards, compare_df, summary_img]
                reload_btn.click(refresh_benchmarks, [run_dd], bench_out)
                run_dd.change(refresh_benchmarks, [run_dd], bench_out)
                demo.load(refresh_benchmarks, [run_dd], bench_out)

            # ================= TAB 5: GUIDE =================
            with gr.Tab("ℹ️ Guide"):
                gr.Markdown(
                    """
### How verdicts are decided
* **Single threshold** — score ≥ threshold ⇒ **DEFECT**, otherwise **NORMAL**.
* **Conformal routing** — score < `T_Normal` ⇒ **NORMAL**; score ≥ `T_Anomaly` ⇒ **DEFECT**;
  anything in between ⇒ **REVIEW** (sent to a human operator).

### Tips
* Press **🎯 Use optimal threshold** to load the F1-optimal threshold saved by `evaluate.py`.
* Lower **Box sensitivity** to reveal fainter defect regions; raise **Minimum region area** to ignore specks.
* Use **Batch Inspection** for whole folders; results export to CSV in `results/exports/`.
* Reviewed labels can be exported from the **Human Review Queue** tab for active learning.

### Workflow
`python train.py` → `python evaluate.py` → `python app.py`
""")

        gr.HTML("<div class='vi-footer'>VisionInspect AI · results are written to <code>results/</code></div>")

    return demo


if __name__ == "__main__":
    demo = build_app()
    print("\n" + "=" * 60)
    print(" 🚀 Launching VisionInspect AI Web Dashboard...")
    print(" Local URL: http://127.0.0.1:7860")
    print("=" * 60 + "\n")
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    demo.launch(server_port=7860, inbrowser=False, share=False, theme=theme)
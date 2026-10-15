"""
src/evaluation/benchmark.py

Limited-data benchmark harness.

Systematically evaluates one or more anomaly detection models at varying
numbers of available defect examples (k = 1, 5, 10, 50, all).
Produces a structured results table and CSV export.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .metrics import full_evaluation_report


class LimitedDataBenchmark:
    """
    Run the limited-data benchmark across:
      - Multiple models
      - Multiple categories
      - Multiple k (few-shot defect counts)

    Usage::
        bench = LimitedDataBenchmark(output_dir="results/benchmark")
        bench.register_model("patchcore", patchcore_model)
        bench.run(category="bottle", dataloader_fn=make_dl, k_values=[1, 5, 10, 50])
        bench.save_results()
        bench.print_summary()
    """

    def __init__(self, output_dir: str = "results/benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Any] = {}
        self.results: List[Dict] = []

    # ------------------------------------------------------------------
    def register_model(self, name: str, model) -> None:
        """Register a model for benchmarking."""
        self.models[name] = model

    # ------------------------------------------------------------------
    def run(
        self,
        category: str,
        train_dataloader_fn,
        test_dataloader,
        k_values: List[int] = [1, 5, 10, 50, -1],
        device: str = "cuda",
        n_trials: int = 3,
        seed: int = 42,
    ) -> None:
        """
        Run benchmarks for all registered models and k values.

        Args:
            category (str): Product category name.
            train_dataloader_fn: Callable(k, seed) → DataLoader for training.
            test_dataloader: Fixed test DataLoader.
            k_values: List of defect counts (-1 = all available).
            device: Compute device.
            n_trials: Number of random trials per k.
            seed: Base random seed.
        """
        device = torch.device(device if torch.cuda.is_available() else "cpu")

        for model_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Model: {model_name}  |  Category: {category}")
            print(f"{'='*60}")

            for k in k_values:
                trial_results = []
                for trial in range(n_trials):
                    trial_seed = seed + trial * 100
                    dl_train = train_dataloader_fn(
                        k=k if k > 0 else None,
                        seed=trial_seed,
                    )

                    # ---- Fit ----
                    if hasattr(model, "fit"):
                        model.fit(dl_train, verbose=False)
                    else:
                        self._train_epoch(model, dl_train, device)

                    # ---- Evaluate ----
                    report = self._evaluate(model, test_dataloader, device)
                    report["model"] = model_name
                    report["category"] = category
                    report["k"] = k
                    report["trial"] = trial
                    trial_results.append(report)
                    print(
                        f"  k={k:>4}  trial={trial}  "
                        f"AUROC={report['image_auroc']:.3f}  "
                        f"PRO={report.get('pro_score', float('nan')):.3f}"
                    )

                # Average over trials
                avg = self._average_trial_results(trial_results, model_name, category, k)
                self.results.append(avg)

    # ------------------------------------------------------------------
    def _train_epoch(self, model, dataloader, device) -> None:
        """Simple single-epoch training for gradient-based models (CAE, StudentTeacher)."""
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for batch in dataloader:
            imgs = batch["image"].to(device)
            opt.zero_grad()
            if hasattr(model, "loss"):
                recon, _ = model(imgs)
                loss = model.loss(imgs, recon)
            else:
                tf, sf = model(imgs)
                loss = model.loss(tf, sf)
            loss.backward()
            opt.step()
        model.eval()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, model, dataloader, device) -> Dict:
        """Collect predictions and compute full evaluation report."""
        image_labels, image_scores = [], []
        gt_masks_list, anomaly_maps_list = [], []
        latencies = []

        for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
            imgs = batch["image"].to(device)
            masks = batch["mask"]
            labels = batch["label"]

            t0 = time.perf_counter()
            amap, score = model.predict(imgs)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) / len(imgs) * 1000)  # ms/image
            image_labels.extend(labels.numpy().tolist())
            image_scores.extend(score.cpu().numpy().tolist())

            # Downscale maps to mask resolution for pixel metrics
            if masks.shape[-2:] != amap.shape[-2:]:
                amap_resized = torch.nn.functional.interpolate(
                    amap, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
            else:
                amap_resized = amap

            gt_masks_list.append(masks.squeeze(1).numpy())
            anomaly_maps_list.append(amap_resized.squeeze(1).cpu().numpy())

        report = full_evaluation_report(
            image_labels=np.array(image_labels),
            image_scores=np.array(image_scores),
            gt_masks=np.concatenate(gt_masks_list),
            anomaly_maps=np.concatenate(anomaly_maps_list),
            latency_ms=float(np.mean(latencies)),
        )
        return report

    # ------------------------------------------------------------------
    @staticmethod
    def _average_trial_results(
        trial_results: List[Dict],
        model_name: str,
        category: str,
        k: int,
    ) -> Dict:
        keys = [k for k in trial_results[0] if isinstance(trial_results[0][k], float)]
        avg = {
            "model": model_name,
            "category": category,
            "k": k,
        }
        for key in keys:
            vals = [r[key] for r in trial_results if not np.isnan(r[key])]
            avg[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
            avg[f"{key}_std"] = float(np.std(vals)) if vals else float("nan")
        return avg

    # ------------------------------------------------------------------
    def save_results(self, filename: str = "benchmark_results.csv") -> Path:
        """Export results to CSV."""
        df = pd.DataFrame(self.results)
        out = self.output_dir / filename
        df.to_csv(out, index=False)
        print(f"\nResults saved to {out}")
        return out

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        """Print a formatted summary table."""
        if not self.results:
            print("No results yet. Run benchmark first.")
            return
        df = pd.DataFrame(self.results)
        cols = ["model", "category", "k", "image_auroc_mean", "pro_score_mean", "latency_ms_mean"]
        available = [c for c in cols if c in df.columns]
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(df[available].to_string(index=False, float_format="%.3f"))
        print("=" * 80)

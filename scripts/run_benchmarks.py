"""
scripts/run_benchmarks.py

Orchestrates the limited defect data experiments (k = 1, 5, 10, 50, all).
Answers the core research question:
"Can anomaly detection and uncertainty-aware human review improve the detection
 of previously unseen manufacturing defects compared with conventional supervised
 classification when labeled defect data is limited?"
"""

import sys
from pathlib import Path
import click
import torch

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data.dataset import MVTecDataset, get_dataloader
from data.transforms import get_train_transforms, get_val_transforms
from models import build_model
from evaluation.benchmark import LimitedDataBenchmark


@click.command()
@click.option("--category", default="bottle", help="MVTec category to benchmark")
@click.option("--data-root", default="data/raw/mvtec", help="Dataset directory")
@click.option("--output-dir", default="results/benchmark", help="Benchmark output directory")
@click.option("--models", default="autoencoder,padim,patchcore", help="Comma-separated model names")
@click.option("--k-values", default="1,5,10,-1", help="Comma-separated defect counts k")
@click.option("--trials", default=3, help="Number of random seed repetitions per k")
@click.option("--device", default="cpu", help="Compute device: cuda or cpu")
def main(category, data_root, output_dir, models, k_values, trials, device):
    """Run limited-data benchmark across multiple anomaly detection models."""
    k_list = [int(x.strip()) for x in k_values.split(",")]
    model_list = [m.strip() for m in models.split(",")]

    print("=" * 70)
    print(" VISIONINSPECT AI — LIMITED TRAINING DATA BENCHMARK")
    print(f" Category:   {category}")
    print(f" Models:     {model_list}")
    print(f" Few-shot k: {k_list}")
    print(f" Trials:     {trials}")
    print("=" * 70)

    bench = LimitedDataBenchmark(output_dir=output_dir)

    for m_name in model_list:
        try:
            m_instance = build_model(m_name, device=device)
            bench.register_model(m_name, m_instance)
        except Exception as e:
            print(f"Warning: Could not build model {m_name}: {e}")

    # Test dataloader
    test_ds = MVTecDataset(root=data_root, category=category, split="test", transform=get_val_transforms())
    test_dl = get_dataloader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    # Train dataloader factory for different k
    def make_train_dl(k=None, seed=42):
        train_ds = MVTecDataset(
            root=data_root,
            category=category,
            split="train",
            transform=get_train_transforms(),
            few_shot_normal=None,
            few_shot_defect=k,
            seed=seed,
        )
        return get_dataloader(train_ds, batch_size=8, shuffle=True, num_workers=0)

    bench.run(
        category=category,
        train_dataloader_fn=make_train_dl,
        test_dataloader=test_dl,
        k_values=k_list,
        device=device,
        n_trials=trials,
    )

    bench.save_results()
    bench.print_summary()


if __name__ == "__main__":
    main()

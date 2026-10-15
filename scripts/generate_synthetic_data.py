"""
scripts/generate_synthetic_data.py

Generates synthetic MVTec-compatible dataset folder structures for all 15 categories:
  Objects: bottle, cable, capsule, hazelnut, metal_nut, pill, screw, toothbrush, transistor, zipper
  Textures: carpet, grid, leather, tile, wood

Usage:
    python scripts/generate_synthetic_data.py --category tile
    python scripts/generate_synthetic_data.py --category capsule
    python scripts/generate_synthetic_data.py --all
"""

import sys
import random
import click
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


# ---------------------------------------------------------------------------
# Procedural Industrial Image Generators
# ---------------------------------------------------------------------------

def generate_base_image(category: str, size=(256, 256), seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img = Image.new("RGB", size, color=(35, 35, 38))
    draw = ImageDraw.Draw(img)

    if category == "bottle":
        draw.rounded_rectangle([70, 70, 185, 230], radius=15, fill=(180, 200, 210), outline=(130, 150, 160), width=2)
        draw.rectangle([105, 30, 150, 75], fill=(160, 180, 190), outline=(120, 140, 150), width=2)
        draw.rectangle([100, 15, 155, 32], fill=(220, 50, 50), outline=(150, 30, 30), width=1)

    elif category == "cable":
        draw.rounded_rectangle([60, 160, 196, 256], radius=8, fill=(50, 52, 55), outline=(30, 30, 32), width=2)
        draw.rounded_rectangle([72, 40, 102, 170], radius=6, fill=(215, 218, 222), outline=(170, 175, 180), width=2)
        draw.rectangle([76, 20, 98, 42], fill=(200, 200, 205), outline=(150, 150, 150), width=1)
        draw.rounded_rectangle([112, 35, 144, 170], radius=6, fill=(0, 130, 200), outline=(0, 90, 150), width=2)
        draw.rectangle([117, 15, 139, 37], fill=(210, 180, 80), outline=(160, 130, 50), width=1)
        draw.rounded_rectangle([154, 45, 184, 170], radius=6, fill=(65, 68, 72), outline=(40, 42, 45), width=2)
        draw.rectangle([159, 25, 179, 47], fill=(190, 190, 195), outline=(130, 130, 130), width=1)

    elif category in ("capsule", "pill"):
        # Two-tone pharmaceutical capsule
        draw.rounded_rectangle([80, 50, 176, 150], radius=35, fill=(230, 40, 40))
        draw.rounded_rectangle([80, 140, 176, 220], radius=35, fill=(240, 240, 245))

    elif category in ("metal_nut", "screw"):
        # Hexagonal industrial metal nut
        cx, cy, r = 128, 128, 70
        points = [
            (cx + r * np.cos(a), cy + r * np.sin(a))
            for a in np.linspace(0, 2 * np.pi, 7)[:-1]
        ]
        draw.polygon(points, fill=(160, 165, 175), outline=(90, 95, 105))
        draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=(35, 35, 38), outline=(90, 95, 105), width=3)

    elif category in ("tile", "wood", "carpet", "leather", "grid"):
        # Structural surface texture
        arr = np.random.normal(160, 15, (size[1], size[0], 3))
        if category == "tile":
            # Grid lines
            arr = np.full((size[1], size[0], 3), 200, dtype=np.float32)
            arr[::64, :] = 80
            arr[:, ::64] = 80
        elif category == "wood":
            # Wood grain bands
            x = np.linspace(0, 10 * np.pi, size[0])
            grain = np.sin(x) * 20
            arr = np.full((size[1], size[0], 3), [170, 120, 70], dtype=np.float32) + grain[None, :, None]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return img

    else:
        # Default industrial component
        draw.rounded_rectangle([60, 60, 196, 196], radius=20, fill=(140, 150, 160), outline=(90, 100, 110), width=3)
        draw.ellipse([100, 100, 156, 156], fill=(70, 80, 90))

    # Add micro-texture noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 3.5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Defect Injection
# ---------------------------------------------------------------------------

def inject_defect(base_img, defect_type="scratch"):
    img = base_img.copy()
    mask = Image.new("L", base_img.size, color=0)
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)
    w, h = base_img.size

    if defect_type in ("scratch", "crack", "cut_wire"):
        x0 = random.randint(int(w * 0.3), int(w * 0.6))
        y0 = random.randint(int(h * 0.3), int(h * 0.6))
        x1 = x0 + random.randint(20, 50)
        y1 = y0 + random.randint(15, 55)
        draw_img.line([(x0, y0), (x1, y1)], fill=(20, 20, 20), width=3)
        draw_mask.line([(x0, y0), (x1, y1)], fill=255, width=4)

    elif defect_type in ("contamination", "hole", "damaged_insulation"):
        cx = random.randint(int(w * 0.35), int(w * 0.65))
        cy = random.randint(int(h * 0.35), int(h * 0.65))
        r = random.randint(8, 16)
        draw_img.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 50, 20))
        draw_mask.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    return img, mask


# ---------------------------------------------------------------------------
# Dataset Builder
# ---------------------------------------------------------------------------

def create_synthetic_mvtec_category(
    base_dir="data/raw/mvtec",
    category="tile",
    n_train=50,
    n_test_good=10,
    n_test_defects=8,
):
    cat_dir = Path(base_dir) / category
    train_good = cat_dir / "train" / "good"
    test_good = cat_dir / "test" / "good"
    gt_dir = cat_dir / "ground_truth"

    defect_type1 = "scratch"
    defect_type2 = "contamination"

    test_def1 = cat_dir / "test" / defect_type1
    test_def2 = cat_dir / "test" / defect_type2
    gt_def1 = gt_dir / defect_type1
    gt_def2 = gt_dir / defect_type2

    for d in [train_good, test_good, test_def1, test_def2, gt_def1, gt_def2]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic {category} dataset at {cat_dir}...")

    # 1. Train Good
    for i in range(n_train):
        img = generate_base_image(category, seed=1000 + i)
        img.save(train_good / f"{i:03d}.png")

    # 2. Test Good
    for i in range(n_test_good):
        img = generate_base_image(category, seed=2000 + i)
        img.save(test_good / f"{i:03d}.png")

    # 3. Test Defect 1 + GT
    for i in range(n_test_defects):
        base = generate_base_image(category, seed=3000 + i)
        defective, mask = inject_defect(base, defect_type1)
        defective.save(test_def1 / f"{i:03d}.png")
        mask.save(gt_def1 / f"{i:03d}_mask.png")

    # 4. Test Defect 2 + GT
    for i in range(n_test_defects):
        base = generate_base_image(category, seed=4000 + i)
        defective, mask = inject_defect(base, defect_type2)
        defective.save(test_def2 / f"{i:03d}.png")
        mask.save(gt_def2 / f"{i:03d}_mask.png")

    print(f"  ✓ {category}: {n_train} train images, {n_test_good} test good, {n_test_defects * 2} defects.")


@click.command()
@click.option("--category", "-c", default="tile", help="Category to generate")
@click.option("--all-categories", "--all", is_flag=True, help="Generate all 15 categories")
@click.option("--n-train", default=50, help="Number of normal train images per category")
@click.option("--base-dir", default="data/raw/mvtec", help="Output directory")
def main(category, all_categories, n_train, base_dir):
    cats = CATEGORIES if all_categories else [category]
    print(f"Generating datasets for: {cats}")
    for c in cats:
        create_synthetic_mvtec_category(base_dir=base_dir, category=c, n_train=n_train)
    print("\n✅ Dataset generation complete!")


if __name__ == "__main__":
    main()

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

    w, h = size
    # Dark industrial background
    img = Image.new("RGB", size, color=(32, 33, 36))
    draw = ImageDraw.Draw(img)

    if category == "bottle":
        # Glass beverage bottle with red cap and neck
        draw.rounded_rectangle([70, 70, 185, 230], radius=15, fill=(180, 200, 210), outline=(130, 150, 160), width=2)
        draw.rectangle([105, 30, 150, 75], fill=(160, 180, 190), outline=(120, 140, 150), width=2)
        draw.rectangle([100, 15, 155, 32], fill=(220, 50, 50), outline=(150, 30, 30), width=1)
        draw.line([(85, 85), (85, 215)], fill=(225, 235, 240), width=3)

    elif category == "cable":
        # Cable with outer jacket and 3 colorful wire strands
        draw.rounded_rectangle([60, 160, 196, 256], radius=8, fill=(50, 52, 55), outline=(30, 30, 32), width=2)
        draw.rounded_rectangle([72, 40, 102, 170], radius=6, fill=(0, 130, 200), outline=(0, 90, 150), width=2)
        draw.rectangle([76, 20, 98, 42], fill=(215, 175, 75), outline=(150, 120, 40), width=1)
        draw.rounded_rectangle([112, 35, 144, 170], radius=6, fill=(220, 222, 226), outline=(170, 175, 180), width=2)
        draw.rectangle([117, 15, 139, 37], fill=(215, 175, 75), outline=(150, 120, 40), width=1)
        draw.rounded_rectangle([154, 45, 184, 170], radius=6, fill=(65, 68, 72), outline=(40, 42, 45), width=2)
        draw.rectangle([159, 25, 179, 47], fill=(215, 175, 75), outline=(150, 120, 40), width=1)

    elif category == "capsule":
        # Two-tone oblong capsule (Crimson Red and White)
        draw.rounded_rectangle([82, 50, 174, 145], radius=35, fill=(225, 38, 45), outline=(180, 25, 30), width=1)
        draw.rounded_rectangle([82, 135, 174, 220], radius=35, fill=(245, 246, 250), outline=(195, 198, 205), width=1)
        draw.line([(96, 68), (96, 200)], fill=(255, 255, 255), width=3)

    elif category == "pill":
        # Round circular tablet with scored dividing line (distinct from capsule)
        cx, cy, r = 128, 128, 62
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(235, 242, 250), outline=(180, 195, 210), width=3)
        draw.ellipse([cx - r + 8, cy - r + 8, cx + r - 8, cy + r - 8], outline=(205, 218, 230), width=1)
        draw.line([(cx, cy - r + 10), (cx, cy + r - 10)], fill=(165, 180, 195), width=3)

    elif category == "metal_nut":
        # Hexagonal metallic nut with threaded center hole
        cx, cy, r = 128, 128, 68
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        points = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
        draw.polygon(points, fill=(160, 168, 178), outline=(100, 108, 118))
        draw.ellipse([cx - 28, cy - 28, cx + 28, cy + 28], fill=(32, 33, 36), outline=(90, 98, 108), width=3)
        draw.ellipse([cx - 24, cy - 24, cx + 24, cy + 24], outline=(70, 76, 85), width=1)
        draw.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], outline=(55, 60, 68), width=1)

    elif category == "screw":
        # Threaded screw with cylindrical shaft, helical thread grooves, and Phillips head
        cx = 128
        draw.ellipse([cx - 38, 28, cx + 38, 70], fill=(175, 180, 190), outline=(110, 115, 125), width=2)
        draw.line([(cx - 18, 49), (cx + 18, 49)], fill=(60, 65, 75), width=3)
        draw.line([(cx, 34), (cx, 64)], fill=(60, 65, 75), width=3)
        draw.rectangle([cx - 15, 68, cx + 15, 210], fill=(145, 150, 160), outline=(100, 105, 115), width=2)
        draw.polygon([(cx - 15, 210), (cx + 15, 210), (cx, 236)], fill=(135, 140, 150), outline=(95, 100, 110))
        for y in range(80, 205, 12):
            draw.line([(cx - 15, y), (cx + 15, y + 6)], fill=(85, 90, 100), width=2)
            draw.line([(cx - 15, y - 1), (cx + 15, y + 5)], fill=(200, 205, 215), width=1)

    elif category == "carpet":
        # Woven looped textile fabric texture
        arr = np.full((h, w, 3), [135, 100, 75], dtype=np.float32)
        yy, xx = np.indices((h, w))
        weave_x = np.sin(xx * (2 * np.pi / 8)) * 18
        weave_y = np.cos(yy * (2 * np.pi / 8)) * 18
        arr[:, :, 0] += weave_x + weave_y
        arr[:, :, 1] += (weave_x * 0.8) + (weave_y * 0.8)
        arr[:, :, 2] += (weave_x * 0.6) + (weave_y * 0.6)
        noise = np.random.normal(0, 12, (h, w, 3))
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    elif category == "grid":
        # Metallic wire mesh grid with welded joints (distinct from carpet)
        arr = np.full((h, w, 3), [28, 30, 35], dtype=np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        for x in range(16, w, 32):
            draw.line([(x, 0), (x, h)], fill=(185, 192, 202), width=5)
            draw.line([(x - 1, 0), (x - 1, h)], fill=(230, 235, 245), width=1)
            draw.line([(x + 2, 0), (x + 2, h)], fill=(90, 95, 105), width=1)
        for y in range(16, h, 32):
            draw.line([(0, y), (w, y)], fill=(175, 182, 192), width=5)
            draw.line([(0, y - 1), (w, y - 1)], fill=(225, 230, 240), width=1)
            draw.line([(0, y + 2), (w, y + 2)], fill=(85, 90, 100), width=1)
        for x in range(16, w, 32):
            for y in range(16, h, 32):
                draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(210, 218, 228))
        return img

    elif category == "leather":
        # Organic pebbled leather texture (saddle brown with cellular grain)
        arr = np.full((h, w, 3), [160, 98, 52], dtype=np.float32)
        yy, xx = np.indices((h, w))
        pebbles = (np.sin(xx * 0.35) * np.cos(yy * 0.35) * 14) + (np.sin(xx * 0.7) * 7)
        arr[:, :, 0] += pebbles
        arr[:, :, 1] += pebbles * 0.65
        arr[:, :, 2] += pebbles * 0.35
        noise = np.random.normal(0, 10, (h, w, 3))
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    elif category == "tile":
        # Glossy ceramic floor tiles with 2x2 grid and dark grout joints
        arr = np.full((h, w, 3), [218, 222, 228], dtype=np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        grout_color = (65, 68, 72)
        draw.line([(0, 128), (w, 128)], fill=grout_color, width=7)
        draw.line([(128, 0), (128, h)], fill=grout_color, width=7)
        draw.line([(0, 124), (w, 124)], fill=(245, 248, 255), width=1)
        draw.line([(124, 0), (124, h)], fill=(245, 248, 255), width=1)
        return img

    elif category == "wood":
        # Wood plank grain with horizontal growth rings
        arr = np.full((h, w, 3), [180, 130, 75], dtype=np.float32)
        yy, xx = np.indices((h, w))
        grain = np.sin(yy * 0.15 + np.sin(xx * 0.03) * 3.0) * 24
        arr[:, :, 0] += grain
        arr[:, :, 1] += grain * 0.75
        arr[:, :, 2] += grain * 0.45
        noise = np.random.normal(0, 7, (h, w, 3))
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    elif category == "hazelnut":
        # Organic hazelnut: acorn-shaped chestnut nut with pale bottom circular scar
        cx, cy = 128, 128
        draw.ellipse([cx - 60, cy - 72, cx + 60, cy + 68], fill=(140, 78, 38), outline=(90, 48, 22), width=2)
        for dx in range(-45, 50, 15):
            draw.arc([cx + dx - 20, cy - 70, cx + dx + 20, cy + 50], start=45, end=135, fill=(115, 60, 28), width=1)
        draw.ellipse([cx - 42, cy + 22, cx + 42, cy + 65], fill=(205, 168, 125), outline=(145, 115, 80), width=2)
        for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
            draw.line([(cx + 25 * np.cos(a), cy + 44 + 14 * np.sin(a)), (cx, cy + 44)], fill=(160, 125, 90), width=1)

    elif category == "toothbrush":
        # Toothbrush handle and head with distinct blue/white bristles
        draw.rounded_rectangle([112, 110, 144, 245], radius=10, fill=(0, 145, 225), outline=(0, 100, 175), width=2)
        draw.rounded_rectangle([116, 40, 140, 120], radius=8, fill=(240, 245, 250), outline=(180, 190, 200), width=2)
        for by in range(48, 105, 10):
            for bx in range(120, 137, 7):
                bristle_color = (0, 120, 220) if (by + bx) % 2 == 0 else (255, 255, 255)
                draw.rectangle([bx, by, bx + 4, by + 6], fill=bristle_color)

    elif category == "transistor":
        # TO-220 3-pin electronic semiconductor component
        cx = 128
        draw.rounded_rectangle([cx - 35, 30, cx + 35, 68], radius=4, fill=(185, 190, 200), outline=(120, 125, 135), width=2)
        draw.ellipse([cx - 10, 40, cx + 10, 60], fill=(32, 33, 36), outline=(110, 115, 125), width=2)
        draw.rounded_rectangle([cx - 40, 65, cx + 40, 145], radius=6, fill=(38, 40, 44), outline=(22, 24, 26), width=2)
        draw.rectangle([cx - 36, 70, cx - 24, 140], fill=(50, 52, 58))
        for lx in [cx - 24, cx, cx + 24]:
            draw.rectangle([lx - 4, 145, lx + 4, 235], fill=(195, 200, 210), outline=(120, 125, 135), width=1)

    elif category == "zipper":
        # Textile zipper tape with interlocking brass teeth and slider tab
        cx = 128
        draw.rectangle([45, 0, cx - 18, h], fill=(42, 45, 52))
        draw.rectangle([cx + 18, 0, 211, h], fill=(42, 45, 52))
        draw.line([(55, 0), (55, h)], fill=(75, 80, 92), width=1)
        draw.line([(201, 0), (201, h)], fill=(75, 80, 92), width=1)
        tooth_color = (215, 175, 75)
        outline_color = (145, 110, 40)
        for y in range(8, h, 12):
            draw.rounded_rectangle([cx - 18, y, cx + 2, y + 6], radius=2, fill=tooth_color, outline=outline_color)
            draw.rounded_rectangle([cx - 2, y + 6, cx + 18, y + 12], radius=2, fill=tooth_color, outline=outline_color)
        draw.rounded_rectangle([cx - 14, 25, cx + 14, 65], radius=5, fill=(180, 185, 195), outline=(100, 105, 115), width=2)
        draw.rectangle([cx - 5, 55, cx + 5, 95], fill=(155, 160, 170), outline=(90, 95, 105))

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

    print(f"  [OK] {category}: {n_train} train images, {n_test_good} test good, {n_test_defects * 2} defects.")


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
    print("\n[SUCCESS] Dataset generation complete!")


if __name__ == "__main__":
    main()

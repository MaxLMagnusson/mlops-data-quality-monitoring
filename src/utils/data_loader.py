"""
Synthetic Data Generator — nuScenes-like Schema
=================================================
Generates synthetic autonomous driving metadata that mirrors the nuScenes
annotation format. Used as the baseline reference dataset for data quality
monitoring.

Schema mirrors nuScenes sample_annotation table:
- Bounding box dimensions (width, height, length)
- 3D translation coordinates (x, y, z)
- Rotation quaternion (w, x, y, z)
- Object category (hierarchical: vehicle.car, human.pedestrian.adult, etc.)
- Visibility level (1-4)
- Number of LiDAR/RADAR points
- Sensor channel metadata
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# nuScenes-like object categories with realistic frequency weights
NUSCENES_CATEGORIES = {
    "vehicle.car": 0.30,
    "vehicle.truck": 0.08,
    "vehicle.bus.rigid": 0.04,
    "vehicle.construction": 0.02,
    "vehicle.trailer": 0.03,
    "human.pedestrian.adult": 0.20,
    "human.pedestrian.child": 0.03,
    "human.pedestrian.construction_worker": 0.02,
    "movable_object.barrier": 0.08,
    "movable_object.trafficcone": 0.06,
    "vehicle.bicycle": 0.04,
    "vehicle.motorcycle": 0.03,
    "movable_object.debris": 0.02,
    "static_object.bicycle_rack": 0.02,
    "animal": 0.01,
    "movable_object.pushable_pullable": 0.02,
}

# Sensor channels matching nuScenes
SENSOR_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "LIDAR_TOP",
]

# Realistic value ranges for nuScenes-like data
VALUE_RANGES = {
    "bbox_width": (0.3, 5.0),       # meters
    "bbox_height": (0.5, 4.0),      # meters
    "bbox_length": (0.5, 12.0),     # meters
    "translation_x": (-50.0, 50.0), # meters from ego vehicle
    "translation_y": (-50.0, 50.0),
    "translation_z": (-3.0, 5.0),
    "num_lidar_pts": (1, 500),
    "num_radar_pts": (0, 50),
    "visibility": (1, 4),           # 1=0-40%, 2=40-60%, 3=60-80%, 4=80-100%
}


def _generate_bbox_dimensions(
    category: str, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Generate realistic bounding box dimensions based on object category."""
    size_profiles = {
        "vehicle.car": {"w": (1.6, 2.1), "h": (1.4, 1.8), "l": (3.8, 5.2)},
        "vehicle.truck": {"w": (2.2, 2.8), "h": (2.5, 3.5), "l": (6.0, 12.0)},
        "vehicle.bus.rigid": {"w": (2.4, 2.8), "h": (3.0, 3.8), "l": (8.0, 12.0)},
        "vehicle.construction": {"w": (2.0, 3.5), "h": (2.0, 4.0), "l": (3.0, 8.0)},
        "vehicle.trailer": {"w": (2.2, 2.8), "h": (2.5, 4.0), "l": (5.0, 15.0)},
        "human.pedestrian.adult": {"w": (0.4, 0.8), "h": (1.5, 2.0), "l": (0.3, 0.7)},
        "human.pedestrian.child": {"w": (0.3, 0.6), "h": (0.8, 1.4), "l": (0.3, 0.5)},
        "human.pedestrian.construction_worker": {"w": (0.5, 0.9), "h": (1.5, 2.0), "l": (0.4, 0.8)},
        "movable_object.barrier": {"w": (0.4, 1.0), "h": (0.8, 1.2), "l": (1.0, 3.0)},
        "movable_object.trafficcone": {"w": (0.3, 0.5), "h": (0.5, 1.0), "l": (0.3, 0.5)},
        "vehicle.bicycle": {"w": (0.4, 0.7), "h": (1.0, 1.5), "l": (1.5, 2.0)},
        "vehicle.motorcycle": {"w": (0.6, 1.0), "h": (1.0, 1.5), "l": (1.8, 2.5)},
    }

    # Get category-specific profile, or use default
    base_category = category.split(".")[0] + "." + category.split(".")[1] if "." in category else category
    profile = size_profiles.get(category, size_profiles.get(base_category, {
        "w": (0.3, 2.0), "h": (0.5, 2.0), "l": (0.3, 3.0)
    }))

    width = rng.uniform(*profile["w"])
    height = rng.uniform(*profile["h"])
    length = rng.uniform(*profile["l"])

    return round(width, 3), round(height, 3), round(length, 3)


def generate_synthetic_baseline(
    n_samples: int = 1000,
    n_scenes: int = 10,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic baseline dataset mimicking nuScenes annotations.

    Parameters
    ----------
    n_samples : int
        Number of annotation rows to generate.
    n_scenes : int
        Number of distinct scenes to distribute annotations across.
    seed : int
        Random seed for reproducibility.
    output_dir : str, optional
        If provided, saves the DataFrame as Parquet to this directory.

    Returns
    -------
    pd.DataFrame
        Synthetic annotation dataset with nuScenes-like schema.
    """
    rng = np.random.Generator(np.random.PCG64(seed))

    categories = list(NUSCENES_CATEGORIES.keys())
    weights = list(NUSCENES_CATEGORIES.values())

    # Generate annotations
    records = []
    for i in range(n_samples):
        category = rng.choice(categories, p=weights)
        w, h, l = _generate_bbox_dimensions(category, rng)
        scene_id = f"scene-{rng.integers(0, n_scenes):04d}"
        sample_token = f"sample-{i:06d}"

        record = {
            "sample_token": sample_token,
            "scene_id": scene_id,
            "category": category,
            "bbox_width": w,
            "bbox_height": h,
            "bbox_length": l,
            "translation_x": round(rng.uniform(*VALUE_RANGES["translation_x"]), 3),
            "translation_y": round(rng.uniform(*VALUE_RANGES["translation_y"]), 3),
            "translation_z": round(rng.uniform(*VALUE_RANGES["translation_z"]), 3),
            "rotation_w": round(rng.uniform(-1.0, 1.0), 4),
            "rotation_x": round(rng.uniform(-0.1, 0.1), 4),
            "rotation_y": round(rng.uniform(-0.1, 0.1), 4),
            "rotation_z": round(rng.uniform(-1.0, 1.0), 4),
            "num_lidar_pts": int(rng.integers(*VALUE_RANGES["num_lidar_pts"])),
            "num_radar_pts": int(rng.integers(*VALUE_RANGES["num_radar_pts"])),
            "visibility": int(rng.integers(*VALUE_RANGES["visibility"])),
            "sensor_channel": rng.choice(SENSOR_CHANNELS),
            "timestamp": int(1_600_000_000 + i * 50_000 + rng.integers(0, 10_000)),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Enforce correct dtypes (important for schema validation)
    float_cols = [
        "bbox_width", "bbox_height", "bbox_length",
        "translation_x", "translation_y", "translation_z",
        "rotation_w", "rotation_x", "rotation_y", "rotation_z",
    ]
    int_cols = ["num_lidar_pts", "num_radar_pts", "visibility", "timestamp"]
    str_cols = ["sample_token", "scene_id", "category", "sensor_channel"]

    for col in float_cols:
        df[col] = df[col].astype("float64")
    for col in int_cols:
        df[col] = df[col].astype("int64")
    for col in str_cols:
        df[col] = df[col].astype("string")

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        parquet_path = output_path / "baseline_metadata.parquet"
        df.to_parquet(parquet_path, index=False, engine="pyarrow")

    return df


def generate_synthetic_images(
    n_images: int = 20,
    width: int = 320,
    height: int = 240,
    output_dir: str = "data/synthetic/baseline_images",
    seed: int = 42,
) -> list[str]:
    """
    Generate small synthetic camera images for the baseline dataset.

    These are simple gradient/noise images to simulate camera frames.
    In production, these would be real nuScenes camera images.

    Parameters
    ----------
    n_images : int
        Number of images to generate.
    width, height : int
        Image dimensions in pixels.
    output_dir : str
        Output directory for generated images.
    seed : int
        Random seed.

    Returns
    -------
    list[str]
        List of generated image file paths.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python-headless is required for image generation. "
            "Install with: pip install opencv-python-headless"
        )

    rng = np.random.Generator(np.random.PCG64(seed))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(n_images):
        # Create a synthetic driving scene image with gradient sky and road
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Sky gradient (top half)
        for y in range(height // 2):
            ratio = y / (height // 2)
            img[y, :] = [
                int(180 + 40 * ratio),  # Blue channel
                int(140 + 60 * ratio),  # Green channel
                int(80 + 40 * ratio),   # Red channel
            ]

        # Road (bottom half) with some noise
        road_noise = rng.integers(60, 100, size=(height // 2, width, 3)).astype(np.uint8)
        img[height // 2:, :] = road_noise

        # Add some random rectangles to simulate objects
        n_objects = rng.integers(1, 5)
        for _ in range(n_objects):
            x1 = int(rng.integers(0, width - 30))
            y1 = int(rng.integers(height // 4, height - 30))
            x2 = int(x1 + rng.integers(20, 80))
            y2 = int(y1 + rng.integers(20, 60))
            color = tuple(int(c) for c in rng.integers(100, 255, size=3))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        filepath = output_path / f"cam_frame_{i:04d}.png"
        cv2.imwrite(str(filepath), img)
        paths.append(str(filepath))

    return paths


if __name__ == "__main__":
    # Generate baseline data for development
    print("Generating synthetic baseline dataset...")
    df = generate_synthetic_baseline(
        n_samples=1000,
        n_scenes=10,
        output_dir="data/synthetic",
    )
    print(f"  → Generated {len(df)} annotations across {df['scene_id'].nunique()} scenes")
    print(f"  → Saved to data/synthetic/baseline_metadata.parquet")
    print(f"  → Schema:\n{df.dtypes}\n")

    print("Generating synthetic camera images...")
    paths = generate_synthetic_images(n_images=20)
    print(f"  → Generated {len(paths)} images")
    print(f"  → Sample: {paths[0]}")

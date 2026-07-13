from pathlib import Path
import sys

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plot_results import plot_example_input_patches


def test_plot_example_input_patches_uses_held_out_manifest(tmp_path):
    rows = []
    for label in (1, 0):
        for index in range(4):
            path = tmp_path / f"{label}_{index}.png"
            Image.new("RGB", (64, 64), color=(40 + label * 120, 80, 120)).save(path)
            rows.append({"path": str(path), "label": label, "split": "test"})

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    output_dir = tmp_path / "figures"

    plot_example_input_patches(manifest, output_dir)

    output = output_dir / "example_input_patches.png"
    assert output.exists()
    assert output.stat().st_size > 0

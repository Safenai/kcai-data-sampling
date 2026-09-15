"""Display helpers for the walkthrough notebook, presentation only.
Drawing an annotation is the dataset reader's job (`DATASET.overlay`)."""

from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.patches import Rectangle

# every column, every value, no truncation
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 10_000)


def hwc(x):
    """CHW float in [0, 1] → HWC, clipped, for imshow."""
    return np.clip(x, 0, 1).transpose(1, 2, 0)


def show(panels, height=4.0):
    """Images side by side. `panels`: list of (title, CHW image)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), height))
    for ax, (title, x) in zip(np.atleast_1d(axes), panels):
        ax.imshow(hwc(x))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_detections(panels, vehicles, height=4.0):
    """`panels`: list of (title, CHW image, detections), each detection a dict with
    `name`, `conf`, `box` = [left, top, right, bottom]. Vehicles red and labelled."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), height))
    for ax, (title, x, detections) in zip(np.atleast_1d(axes), panels):
        ax.imshow(hwc(x))
        ax.axis("off")
        ax.set_title(f"{title}, {len(detections)} detections", fontsize=10)
        for d in detections:
            left, top, right, bottom = d["box"]
            vehicle = d["name"] in vehicles
            ax.add_patch(Rectangle((left, top), right - left, bottom - top, fill=False,
                                   edgecolor="red" if vehicle else "cyan", linewidth=1.2))
            if vehicle:
                ax.text(left, top - 4, f"{d['name']} {d['conf']:.2f}", color="white", fontsize=7,
                        backgroundcolor="black")
    plt.tight_layout()
    plt.show()


def rows_table(records, caption=None):
    """Output rows as a DataFrame, every column; `None` shown as `, `."""
    frame = pd.DataFrame([asdict(r) for r in records]).astype(object).fillna(", ")
    if caption:
        print(caption)
    display(frame)

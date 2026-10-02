"""Display helpers for the walkthrough notebook, presentation only.

Drawing an annotation is the notebook's job here: the interface never reads
the interface batch's pixel data, and the parquet loader reads frames only.
"""

import base64
import io
import json

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from PIL import Image

# every column, every value, no truncation
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 10_000)


def _jpeg_bytes(x, width=640):
    """Encode one HWC uint8 image as JPEG bytes, downscaled for display.

    Args:
        x: ``(H, W, 3|4)`` uint8 array; an alpha channel is dropped.
        width: Display width in pixels; larger images are downscaled.

    Returns:
        The JPEG-encoded bytes of the image.
    """
    array = np.ascontiguousarray(x[..., :3] if x.shape[2] == 4 else x)
    img = Image.fromarray(array)
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def show(panels, width=640):
    """Show images side by side, one JPEG row with captions.

    Args:
        panels: List of ``(title, image)`` tuples; the image is HWC uint8.
        width: Display width of each panel, in pixels.
    """
    figures = []
    for title, x in panels:
        data = base64.b64encode(_jpeg_bytes(x, width)).decode()
        figures.append(
            "<figure style='margin:0 8px'>"
            f"<img src='data:image/jpeg;base64,{data}'/>"
            f"<figcaption style='font-size:10px'>{title}</figcaption>"
            "</figure>"
        )
    display(HTML("<div style='display:flex;align-items:flex-start;flex-wrap:wrap'>" + "".join(figures) + "</div>"))


def rows_table(outputs, caption=None):
    """Show outputs as a DataFrame, every field but the bitmap.

    Args:
        outputs: The ``Output`` rows to show.
        caption: A caption printed above the table, or ``None``.
    """
    rows = []
    for o in outputs:
        row = o.row()
        row["params"] = json.dumps(row["params"], sort_keys=True, default=str)
        rows.append(row)
    frame = pd.DataFrame(rows).astype(object).fillna("-")
    if caption:
        print(caption)
    display(frame)

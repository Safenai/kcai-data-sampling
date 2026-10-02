#!/usr/bin/env python3
"""Download the ten comma10k frames the tests and the notebook use, into
`examples/data/comma10k_sample/` (git-ignored).

    python scripts/fetch_comma10k_sample.py

comma10k: https://github.com/commaai/comma10k, MIT licence.
"""

import urllib.request
from pathlib import Path

COMMIT = "6c205fe4c43cc53b2b1befafb1060d0606555027"
RAW = f"https://raw.githubusercontent.com/commaai/comma10k/{COMMIT}"
NAMES = [
    "00009_f_20f59690c1da9379_2021-02-23--09-20-54_21_334.png",
    "00012_f_a1fc603d9a8ddfc4_2021-02-14--14-05-48_71_125.png",
    "00026_f_d908422f324193fe_2021-03-04--16-39-08_35_286.png",
    "00035_f_20f59690c1da9379_2021-02-20--13-31-14_50_347.png",
    "00040_f_5352b3c0dcecc48d_2021-02-09--15-46-00_9_366.png",
    "00055_f_a1fc603d9a8ddfc4_2021-02-14--14-05-48_85_93.png",
    "00072_f_20f59690c1da9379_2021-02-22--08-21-23_4_672.png",
    "00076_f_e2a273d7e6eecec2_2021-03-03--16-46-12_7_1160.png",
    "00103_f_a1fc603d9a8ddfc4_2021-02-14--14-05-48_37_929.png",
    "00118_f_e53b4945e2d249ba_2021-03-01--17-54-03_34_245.png",
]
OUT = Path(__file__).resolve().parents[1] / "examples" / "data" / "comma10k_sample"


def write_sidecar() -> None:
    """Write the ``samples.parquet`` sidecar for the fetched frames.

    One row per image: ``id`` (file stem), ``path`` (relative to the sample
    root, resolved by ``sample_path.prefix``), ``height`` and ``width``
    (real image sizes, so no ``image_shape`` is needed).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    rows = []
    for name in sorted((OUT / "imgs").glob("*.png")):
        w, h = Image.open(name).size
        rows.append((name.stem, f"imgs/{name.name}", h, w))
    table = pa.table(
        {
            "id": [r[0] for r in rows],
            "path": [r[1] for r in rows],
            "height": [r[2] for r in rows],
            "width": [r[3] for r in rows],
        }
    )
    pq.write_table(table, OUT / "samples.parquet")
    print(f"sidecar {OUT / 'samples.parquet'} with {len(rows)} rows")


def main() -> None:
    for remote, local in (("imgs2", "imgs"), ("masks2", "masks")):
        (OUT / local).mkdir(parents=True, exist_ok=True)
        for name in NAMES:
            path = OUT / local / name
            if path.exists():
                continue
            with urllib.request.urlopen(f"{RAW}/{remote}/{name}", timeout=120) as r:
                path.write_bytes(r.read())
            print(f"  {local}/{name[:12]}…")
    write_sidecar()
    print(f"{len(NAMES)} frames and masks in {OUT}")


if __name__ == "__main__":
    main()
# datasets/parquet_dataset.py
from dataclasses import dataclass, asdict
import io
import json
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

# IOConfig : configuration for input/output columns in the dataset.
# TODO: 
@dataclass
class IOConfig:
    """Configuration for input/output columns in the dataset."""
    id_col: str = "sample_id"
    class_col: str = "class_name"
    image_bytes_col: str = "image_bytes"
    split_col: str = "split"
    columns: tuple = None  

class ParquetDataset(Dataset):


    _cache = {}
    def __init__(self, parquet_path, splitter=None, io_cfg: IOConfig = IOConfig(), 
                 split="train", transform=None, save_config_path=None, config_name="default"):

        self.path = parquet_path
        self.io = io_cfg
        self.splitter = splitter
        self.split = split
        self.transform = transform
        self.save_config_path = save_config_path
        self.config_name = config_name

        cache_key = (parquet_path, config_name, splitter is not None)
        if cache_key not in ParquetDataset._cache:
            tbl = self._load_minimal()

            if self.splitter is not None:
                tbl = self.splitter.apply(tbl)  # ⇐ strategy plugged here
            else:
                # check that split column exists
                if self.io.split_col not in tbl.column_names:
                    raise ValueError(f"No splitter provided and column '{self.io.split_col}' not found in Parquet file")

            le = self._build_label_encoder(tbl)
            ParquetDataset._cache[cache_key] = (tbl, le)
            self._persist_meta(tbl, le)

        self.table, self.le = ParquetDataset._cache[cache_key]

        mask = pa.compute.equal(self.table[self.io.split_col], self.split)
        self.view = self.table.filter(mask)

        self._img = self.view[self.io.image_bytes_col]
        labels_np = np.array(self.view[self.io.class_col].to_pylist(), dtype=object)
        self._y = self.le.transform(labels_np)


        self.classes = self.le.classes_.tolist()  # for torchvision compatibility
        self.targets = self._y.tolist()  # for pruner compatibility

    def _load_minimal(self) -> pa.Table:
        necessary = {self.io.id_col, self.io.class_col, self.io.image_bytes_col}

        if self.splitter is None:
            necessary.add(self.io.split_col)

        cols = list(self.io.columns) if self.io.columns else list(necessary)
        return pq.read_table(self.path, columns=cols, memory_map=True)

    def _build_label_encoder(self, table: pa.Table):
        le = LabelEncoder()
        le.fit(np.unique(np.array(table[self.io.class_col].to_pylist(), dtype=object)))
        return le

    def _persist_meta(self, table: pa.Table, le: LabelEncoder):
        if not self.save_config_path:
            return

        # calculate split statistics
        split_counts = {}
        for s in ["train", "val", "test"]:
            count = pa.compute.sum(pa.compute.equal(table[self.io.split_col], s)).as_py()
            split_counts[s] = int(count) if count is not None else 0

        meta = {
            "config_name": self.config_name,
            "io": asdict(self.io),
            "split_counts": split_counts,
            "classes": le.classes_.tolist(),
        }

        obj = {}
        if os.path.exists(self.save_config_path):
            try:
                with open(self.save_config_path, "r") as f:
                    obj = json.load(f)
            except Exception:
                obj = {}

        obj[self.config_name] = meta

        # Save
        with open(self.save_config_path, "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def __len__(self):
        """Return the number of samples in the split."""
        return len(self.view)

    def __getitem__(self, idx: int):

        row = self.view.slice(idx, 1)
        buf = row[self.io.image_bytes_col][0].as_buffer()

        img = Image.open(io.BytesIO(buf.to_pybytes()))
        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, int(self._y[idx])

    def get_class_names(self):
        return self.le.classes_.tolist()

    def get_split_info(self):
        if hasattr(self, 'table'):
            split_counts = {}
            for s in ["train", "val", "test"]:
                count = pa.compute.sum(pa.compute.equal(self.table[self.io.split_col], s)).as_py()
                split_counts[s] = int(count) if count is not None else 0
            return split_counts
        return None

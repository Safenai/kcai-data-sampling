# splitters/stratified_hash.py
import numpy as np
import pyarrow as pa
from collections import defaultdict
from .hash import _hash16, _frac_from_hex
from .base import Splitter

class StratifiedHashSplitter(Splitter):

    def __init__(self, by_cols=("class_name",), image_bytes_col="image_bytes", 
                 ratios=(0.7, 0.2, 0.1), out_col="split"):
        """
        Args:
            TODO : Prendre aussi les images des paths pour le split.
            by_cols: Column(s) for stratification (tuple or list)
            image_bytes_col: 
            ratios: tuple (train_ratio, val_ratio, test_ratio) - must sum to 1.0
            out_col: name of the output column for splits
        """
        assert abs(sum(ratios) - 1.0) < 1e-9, f"Ratios must sum to 1.0, got {sum(ratios)}"
        self.by_cols = by_cols if isinstance(by_cols, (list, tuple)) else (by_cols,)
        self.image_bytes_col = image_bytes_col
        self.ratios = ratios
        self.out_col = out_col

    def apply(self, table: pa.Table) -> pa.Table:

        tbl = table
        # TODO : prendre un nom de colonne pour les hashes.
        if "image_hash" not in tbl.column_names:
            hashes = [_hash16(tbl[self.image_bytes_col][i].as_buffer()) for i in range(len(tbl))]
            tbl = tbl.append_column("image_hash", pa.array(hashes, type=pa.string()))

        keys_np = list(zip(*[tbl[c].to_pylist() for c in self.by_cols]))
        idx_by_key = defaultdict(list)
        for i, k in enumerate(keys_np):
            idx_by_key[k].append(i)

        chunks = []
        thr1, thr2 = self.ratios[0], self.ratios[0] + self.ratios[1]
        
        for _, idxs in idx_by_key.items():
            g = tbl.take(pa.array(idxs, type=pa.int64()))
            
            xs = np.array([_frac_from_hex(h) for h in g["image_hash"].to_pylist()], dtype=np.float64)
            split = np.where(xs < thr1, "train", np.where(xs < thr2, "val", "test"))
            
            chunks.append(g.append_column(self.out_col, pa.array(split, type=pa.string())))

        return pa.concat_tables(chunks, promote=True)

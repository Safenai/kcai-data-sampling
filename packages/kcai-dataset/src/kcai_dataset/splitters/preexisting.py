# splitters/preexisting.py
import pyarrow as pa
import pyarrow.compute as pc
from .base import Splitter

# PreExistingSplitter : utiliser une colonne existante pour le split.
class PreExistingSplitter(Splitter):
    def __init__(self, split_col: str = "split"):

        self.split_col = split_col

    def apply(self, table: pa.Table) -> pa.Table:
        if self.split_col not in table.column_names:
            raise ValueError(f"Column '{self.split_col}' not found.")
        
        col = table[self.split_col]
        mapped = pc.if_else(pc.equal(col, "validation"), pa.scalar("val"), col)
        col_index = table.column_names.index(self.split_col)
        return table.set_column(col_index, self.split_col, mapped)

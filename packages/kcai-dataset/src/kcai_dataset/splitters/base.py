# splitters/base.py
from abc import ABC, abstractmethod
import pyarrow as pa

class Splitter(ABC):
    """Returns a Table with a 'split' column (train/val/test).
    or train/val/eval
    """
    
    @abstractmethod
    def apply(self, table: pa.Table) -> pa.Table:
        """
        Apply the splitting strategy to the PyArrow table.
        
        Args:
            table: Input PyArrow table
            
        Returns:
            PyArrow table with 'split' column added or modified
        """
        ...

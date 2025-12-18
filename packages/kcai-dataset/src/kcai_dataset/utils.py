"""
Utility functions for kc-dataset.
"""
import uuid
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq


def add_uuid_to_parquet(
    input_path: str, 
    output_path: Optional[str] = None, 
    uuid_column: str = "sample_uuid",
    overwrite_existing: bool = False
):
    """
    Add a unique UUID to each row in a parquet file.
    
    Uses PyArrow (already a dependency) so no additional packages needed.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file (if None, overwrites input)
        uuid_column: Name of the UUID column to add (default: "sample_uuid")
        overwrite_existing: If True, overwrite existing UUID column without asking.
                           If False and column exists, raise ValueError.
    
    Returns:
        Path to the output file
    
    Raises:
        ValueError: If UUID column exists and overwrite_existing is False
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
    
    # read parquet file
    table = pq.read_table(input_path)
    
    # check if UUID column already exists
    if uuid_column in table.column_names:
        if not overwrite_existing:
            raise ValueError(
                f"Column '{uuid_column}' already exists. "
                f"Set overwrite_existing=True to overwrite it."
            )
        # remove existing column
        col_idx = table.column_names.index(uuid_column)
        table = table.remove_column(col_idx)
    
    # generate UUIDs for each row
    num_rows = len(table)
    uuids = [str(uuid.uuid4()) for _ in range(num_rows)]
    
    # verify uniqueness (should never fail, but good to check)
    if len(set(uuids)) != num_rows:
        raise RuntimeError("Failed to generate unique UUIDs!")
    
    # create UUID column as PyArrow array
    uuid_array = pa.array(uuids, type=pa.string())
    
    # Add UUID column - we'll create a new table with UUID first
    # Get all existing column names
    existing_names = list(table.column_names)
    
    # Create new table with UUID first, then existing columns
    new_columns = {uuid_column: uuid_array}
    for name in existing_names:
        new_columns[name] = table[name]
    
    new_table = pa.table(new_columns)
    
    # Reorder to put UUID first
    column_order = [uuid_column] + existing_names
    new_table = new_table.select(column_order)
    
    # Save
    pq.write_table(new_table, output_path)
    
    return str(output_path)

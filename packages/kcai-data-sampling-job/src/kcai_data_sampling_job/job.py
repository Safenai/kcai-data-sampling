"""The orchestration job: selections in, metadata rows and payloads out.

Streaming contract: the job decodes one
``load_batch_size`` chunk into an in-memory selection, runs each configured
transformation over the chunk through the runner, encodes the payloads (in
memory, buffered by the writers) as it goes, buffers the metadata rows, and
drops the chunk. Any batching here
changes nothing in the output (unary partition invariance; n-ary pairing is
chunk-local).

A row per output: every transformation is applied independently to the same
chunk, and each output is its own row — there is no chain and no intermediate
stage between the source sample and the output.
"""

import json
import logging
from typing import Any

from kcai_data_sampling_core.api.output import Output
from kcai_data_sampling_core.api.selection import Batch
from kcai_data_sampling_core.api.transformation import Transformation
from kcai_data_sampling_core.models.global_ import ErrorsConfig
from kcai_data_sampling_core.utils.matching import resolve_include_exclude
from kcai_data_sampling_core.utils.runner import TransformationRunner
from kcai_data_sampling_job.dataloaders import DataLoader, DataSelection as SourceSelection

logger = logging.getLogger(__name__)

#: The generator columns of the ledger; pass-through source columns may not
#: overwrite them (the source row id already lives in ``parent_id``).
GENERATOR_COLUMNS = (
    "selection",
    "dataloader",
    "parent_id",
    "id",
    "algorithm",
    "family",
    "arity",
    "reversible",
    "params",
    "seed",
    "tool_model",
    "target_model",
    "artifact",
)


class SamplingJob:
    """Orchestrates loading, transforming, and writing one job run.

    Attributes:
        dataloaders: Map of loader name → loader.
        transformations: The list of transformations, each applied
            independently to the same chunk.
        payload_writer: Writer that encodes each output to a payload file.
        ledger_writer: Writer that persists the metadata-only rows.
        errors: The merged error policy for the transformations interface.
        transform_batch_size: Target ``(B, *sample)`` handed to apply.
    """

    def __init__(
        self,
        dataloaders: dict[str, DataLoader],
        transformations: list[Transformation],
        payload_writer: Any | None = None,
        ledger_writer: Any | None = None,
        errors: ErrorsConfig | None = None,
        progress_bar: bool = True,
        transform_batch_size: int | None = None,
        include_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
    ):
        """Build the job.

        Args:
            dataloaders: Loader instances to run.
            transformations: Ordered transformations.
            payload_writer: The payload (images) writer, or ``None``.
            ledger_writer: The ledger (parquet) writer, or ``None``.
            errors: Merged interface error policy.
            progress_bar: Show tqdm progress bars.
            transform_batch_size: Target transform batch; ``None`` defaults
                to each loader's ``load_batch_size``.
            include_columns: Source columns to pass through to the ledger
                (wildcards ok); ``None`` passes nothing unless
                ``exclude_columns`` is set.
            exclude_columns: Source columns withheld from the ledger; wins
                over ``include_columns``.
        """
        self.dataloaders = dataloaders
        self.transformations = transformations
        self.payload_writer = payload_writer
        self.ledger_writer = ledger_writer
        self.errors = errors or ErrorsConfig()
        self.progress_bar = progress_bar
        self.transform_batch_size = transform_batch_size
        self.include_columns = include_columns
        self.exclude_columns = exclude_columns

    def run(self) -> dict[str, int]:
        """Run the job over every selection of every loader.

        Returns:
            A summary mapping each selection name to its number of emitted
            output rows.

        Raises:
            RuntimeError: If a selection's failure rate exceeds
                ``errors.max_failure_rate``.
        """
        summary: dict[str, int] = {}
        from tqdm import tqdm

        selections = [s for loader in self.dataloaders.values() for s in loader.get_selections()]

        selection_iter = (
            tqdm(selections, desc="selection", position=0) if self.progress_bar else selections
        )

        for selection in selection_iter:
            selection.bootstrap(None)

            batches_iter = (
                tqdm(
                    selection,
                    desc="batches",
                    position=1,
                    leave=False,
                    total=selection.get_nb_batches(),
                )
                if self.progress_bar
                else selection
            )

            row_count = 0
            for batch in batches_iter:
                row_count += self._process_batch(selection, batch)

            self._check_failure_rate(selection)

            if self.payload_writer is not None:
                self.payload_writer.flush()
            if self.ledger_writer is not None:
                self.ledger_writer.flush()
            summary[selection.name] = row_count
            logger.info(
                "Selection '%s': %s output rows, %s payload files",
                selection.name,
                row_count,
                self.payload_writer and "enabled" or "disabled",
            )
        return summary

    def _process_batch(self, selection: SourceSelection, batch: Batch) -> int:
        """Run every transformation independently over one decoded batch.

        The loop is **sample-major**: for each source row, every configured
        transformation (one instance per swept value, in config order) is
        applied, chunked into groups of ``transform_batch_size`` so only a
        bounded number of variants of one row lives in memory at a time.
        Non-swept transformations are a single-variant case of the same loop.

        Args:
            selection: The source selection the batch came from.
            batch: The decoded batch (batch = row).

        Returns:
            The number of output rows emitted for this chunk.
        """
        runner = TransformationRunner(batch)

        chunk = self.transform_batch_size or len(self.transformations) or 1
        pending: list[dict[str, Any]] = []
        count = 0

        passed = self._passed_rows(batch)

        for i in range(len(batch)):
            row = batch.row(i)
            for start in range(0, len(self.transformations), chunk):
                group = self.transformations[start : start + chunk]
                for transformation in group:
                    for output in runner.run(transformation, batch=row):
                        artifact = None
                        if self.payload_writer is not None:
                            artifact = self.payload_writer.add_payload(selection.name, output)
                        pending.append(self._row(selection.name, selection.dataset, output, artifact, passed[i]))
                        count += 1

        if self.ledger_writer is not None and pending:
            self.ledger_writer.add_rows(selection.name, pending)
        return count

    def _passed_rows(self, batch: Batch) -> list[dict[str, Any]]:
        """The ledger pass-through source columns of a batch, per row.

        ``outputs.include``/``exclude`` select source columns arrow-natively
        from the batch's arrow table; ``exclude`` wins over ``include`` and
        generator columns are never overwritten. With neither set, nothing is
        passed through.

        Args:
            batch: The decoded batch.

        Returns:
            One dict per row holding its selected source values.
        """
        available = batch.columns.column_names
        if self.include_columns is not None or self.exclude_columns is not None:
            names = resolve_include_exclude(self.include_columns, self.exclude_columns, available)
            names = [c for c in names if c in available and c not in GENERATOR_COLUMNS]
        else:
            names = []
        if not names:
            return [{} for _ in range(len(batch))]
        return batch.columns.select(names).to_pylist()

    @staticmethod
    def _row(
        selection_name: str,
        dataset: str,
        output: Output,
        artifact: str | None,
        passed: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build one metadata-only ledger row from an output.

        The 13 generator columns are fixed; ``passed`` source values are
        appended without overwriting them (the source row id already lives in
        ``parent_id``).

        Args:
            selection_name: The selection the row belongs to.
            dataset: The loader that assembled the selection.
            output: The generated output.
            artifact: The payload artifact file name, or ``None`` in
                trace-only mode.
            passed: One row's selected source values, or ``None``.

        Returns:
            The row dictionary (the ledger columns).
        """
        row = {
            "selection": selection_name,
            "dataloader": dataset,
            "parent_id": output.parent_id,
            "id": output.id,
            "algorithm": output.algorithm,
            "family": output.family,
            "arity": output.arity,
            "reversible": output.reversible,
            "params": json.dumps(output.params, sort_keys=True, default=str),
            "seed": output.seed,
            "tool_model": output.tool_model,
            "target_model": output.target_model,
            "artifact": artifact,
        }
        if passed:
            row.update({k: v for k, v in passed.items() if k not in row})
        return row

    def _check_failure_rate(self, selection: SourceSelection) -> None:
        """Abort when the selection's failure rate exceeds the policy.

        Args:
            selection: The selection just processed.

        Raises:
            RuntimeError: If failures/samples exceed ``max_failure_rate``.
        """
        failures = int(getattr(selection, "failures", 0))
        total = int(getattr(selection, "nb_samples", 0))
        if total and failures / total > self.errors.max_failure_rate:
            raise RuntimeError(
                f"selection '{selection.name}': {failures}/{total} rows failed, above the "
                f"max_failure_rate {self.errors.max_failure_rate}"
            )
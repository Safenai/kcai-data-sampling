"""The orchestration job: selections in, metadata rows and payloads out.

Streaming contract: the job decodes one
``load_batch_size`` chunk into an in-memory selection, runs each configured
transformation over the chunk through the runner, writes the payloads as it
goes, buffers the metadata rows, and drops the chunk. Any batching here
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
from kcai_data_sampling_core.api.selection import DataSelection, Sample
from kcai_data_sampling_core.api.transformation import Transformation
from kcai_data_sampling_core.models.global_ import ErrorsConfig
from kcai_data_sampling_core.utils.runner import TransformationRunner
from kcai_data_sampling_job.dataloaders import DataLoader, DataSelection as SourceSelection

logger = logging.getLogger(__name__)


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
        """
        self.dataloaders = dataloaders
        self.transformations = transformations
        self.payload_writer = payload_writer
        self.ledger_writer = ledger_writer
        self.errors = errors or ErrorsConfig()
        self.progress_bar = progress_bar
        self.transform_batch_size = transform_batch_size

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

    def _process_batch(self, selection: SourceSelection, batch: list[Sample]) -> int:
        """Run every transformation independently over one decoded chunk.

        Args:
            selection: The source selection the chunk came from.
            batch: The decoded sample chunk.

        Returns:
            The number of output rows emitted for this chunk.
        """
        memory_selection = DataSelection(
            name=selection.name,
            dataset=selection.dataset,
            samples=batch,
            sample_axes=selection.sample_axes,
            value_range=selection.value_range,
        )

        pending: list[dict[str, Any]] = []
        count = 0

        for transformation in self.transformations:
            runner = TransformationRunner(memory_selection)
            outputs = runner.run(
                transformation,
                samples=batch,
                batch_size=self.transform_batch_size,
            )
            for output in outputs:
                artifact = None
                if self.payload_writer is not None:
                    artifact = self.payload_writer.write_payload(selection.name, output)
                pending.append(self._row(selection.name, selection.dataset, output, artifact))
                count += 1

        if self.ledger_writer is not None and pending:
            self.ledger_writer.add_rows(selection.name, pending)
        return count

    @staticmethod
    def _row(selection_name: str, dataset: str, output: Output, artifact: str | None) -> dict[str, Any]:
        """Build one metadata-only ledger row from an output.

        Args:
            selection_name: The selection the row belongs to.
            dataset: The loader that assembled the selection.
            output: The generated output.
            artifact: The payload artifact file name, or ``None`` in
                trace-only mode.

        Returns:
            The row dictionary (the ledger columns).
        """
        return {
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
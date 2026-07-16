"""Registry of the DANDI datasets used by NeuroMysteryBench.

Pinning the dataset *version* guarantees reproducible ground truth. Versions
marked ``None`` still need to be pinned to a published release before that
category's questions are finalized (tracked in the implementation plan).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    dandi_id: str
    version: str | None
    name: str
    category: int
    modality: str


DATASETS: dict[str, Dataset] = {
    "000458": Dataset(
        dandi_id="000458",
        version="0.230317.0039",
        name="Simultaneous EEG + extracellular ephys + cortical imaging",
        category=3,
        modality="Multi-modal (EEG, LFP, spikes)",
    ),
    "000035": Dataset(
        dandi_id="000035",
        version="0.211014.0808",
        name="Patch-seq mouse motor cortex (Gouwens et al.)",
        category=2,
        modality="Patch-seq (ephys + transcriptomics)",
    ),
    "000409": Dataset(
        dandi_id="000409",
        version="0.260309.1324",
        name="IBL Brain Wide Map",
        category=1,
        modality="Neuropixels extracellular ephys",
    ),
    "000003": Dataset(
        dandi_id="000003",
        version=None,  # TODO: pin a published version before finalizing Category 4
        name="Hippocampal granule cells and mossy cells (Senzai & Buzsaki)",
        category=4,
        modality="Tetrode ephys + LFP",
    ),
    "000552": Dataset(
        dandi_id="000552",
        version=None,  # TODO: pin a published version before finalizing Category 5
        name="Hippocampus embryonic birthdate (Bhaskaran et al.)",
        category=5,
        modality="Tetrode ephys",
    ),
}

# The pilot dataset for the vertical slice (implementation plan Stage 1-7).
PILOT_DANDI_ID = "000458"


def get_dataset(dandi_id: str) -> Dataset:
    try:
        return DATASETS[dandi_id]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset {dandi_id!r}; known: {sorted(DATASETS)}") from exc


def require_version(dandi_id: str) -> str:
    """Return the pinned version for a dataset, or raise if not yet pinned."""
    ds = get_dataset(dandi_id)
    if ds.version is None:
        raise ValueError(
            f"Dataset {dandi_id} ({ds.name}) has no pinned version yet; "
            "pin a published release in data/datasets.py before use."
        )
    return ds.version

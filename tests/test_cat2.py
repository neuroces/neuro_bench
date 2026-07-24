"""Offline tests for Category-2 (patch-seq) resolvers and icephys tools."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from agent.tools import ToolSession
from benchmark import ground_truth


def make_icephys_stub(genotype="Pvalb-Cre/wt; Ai9/wt"):
    """Synthetic current-clamp cell: sweeps at -50/+50/+100/+200 pA with 0/0/1/5 spikes."""
    fs = 25000.0
    n = 1000

    def sweep(inj_pA, n_spikes):
        stim = np.zeros(n)
        stim[100:900] = inj_pA
        v = np.full(n, -65.0)
        for k in range(n_spikes):
            v[150 + k * 20] = 20.0  # single-sample upward crossing of -10 mV
        return v, stim

    acq, sti = {}, {}
    for i, (inj, ns) in enumerate([(-50, 0), (50, 0), (100, 1), (200, 5)]):
        v, stim = sweep(inj, ns)
        acq[f"CurrentClampSeries{i:03d}"] = SimpleNamespace(data=v, conversion=1.0, rate=fs)
        sti[f"CurrentClampStimulusSeries{i:03d}"] = SimpleNamespace(data=stim, conversion=1.0, rate=fs)

    subject = SimpleNamespace(subject_id="m", species="Mus musculus", sex="M", genotype=genotype)
    return SimpleNamespace(
        acquisition=acq,
        stimulus=sti,
        subject=subject,
        intervals={},
        units=None,
        identifier="ic-stub",
        session_description="Current clamp square",
    )


# --- resolvers ---------------------------------------------------------------


def test_cell_subtype_from_genotype():
    assert ground_truth.cell_subtype(make_icephys_stub("Pvalb-Cre/wt; Ai9/wt")) == "Pvalb"
    assert ground_truth.cell_subtype(make_icephys_stub("Sst-IRES-Cre/wt; Ai9/wt")) == "Sst"


def test_cell_class_inhibitory():
    assert ground_truth.cell_class(make_icephys_stub("Sst-IRES-Cre/wt; Ai9/wt")) == "inhibitory"


def test_rheobase_is_min_positive_spiking_step():
    assert ground_truth.rheobase(make_icephys_stub()) == 100.0


def test_firing_rate_at_current():
    # 200 pA sweep has 5 spikes over an 0.032 s pulse -> ~156 Hz.
    rate = ground_truth.firing_rate_at_current(make_icephys_stub(), current_pA=200.0)
    assert rate == pytest.approx(156.25, abs=1.0)


def test_icephys_sweeps_structure():
    sweeps = ground_truth.icephys_sweeps(make_icephys_stub())
    assert [s["injected_pA"] for s in sweeps] == [-50.0, 50.0, 100.0, 200.0]
    assert [s["n_spikes"] for s in sweeps] == [0, 0, 1, 5]


# --- icephys tools -----------------------------------------------------------


def test_list_and_get_current_clamp_sweep():
    s = ToolSession(nwb=make_icephys_stub())
    listed = s.list_current_clamp_sweeps()
    assert listed["n_sweeps"] == 4
    sweep = s.get_current_clamp_sweep(2)
    assert sweep["injected_pA"] == 100.0
    assert sweep["n_spikes"] == 1


def test_describe_recording_redacts_subject():
    s = ToolSession(nwb=make_icephys_stub(), redaction=["subject"])
    assert "subject" not in s.describe_recording()
    s2 = ToolSession(nwb=make_icephys_stub(), redaction=[])
    # even unredacted, genotype is never exposed
    assert "genotype" not in s2.describe_recording().get("subject", {})

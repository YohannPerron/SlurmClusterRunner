from __future__ import annotations

import pytest

from src.sweep import SweepConfirmationRequired, parse_sweep, split_bracket_aware


def test_split_bracket_aware_preserves_bracketed_commas():
    assert split_bracket_aware("1e-3,1e-4") == ["1e-3", "1e-4"]
    assert split_bracket_aware("[1,2,3]") == ["[1,2,3]"]
    assert split_bracket_aware("checkpoint_a.ckpt,checkpoint_b.ckpt") == [
        "checkpoint_a.ckpt",
        "checkpoint_b.ckpt",
    ]


def test_hydra_only_sweep():
    plan = parse_sweep(["lr=1e-3,1e-4", "model=resnet,vgg"])

    assert [axis.name for axis in plan.variable_axes] == ["lr", "model"]
    assert plan.jobs[0].tokens == ["lr=1e-3", "model=resnet"]
    assert [job.hydra_overrides for job in plan.jobs] == [
        ["lr=1e-3", "model=resnet"],
        ["lr=1e-3", "model=vgg"],
        ["lr=1e-4", "model=resnet"],
        ["lr=1e-4", "model=vgg"],
    ]
    assert all(job.positional_args == [] for job in plan.jobs)


def test_positional_only_sweep_preserves_order():
    plan = parse_sweep(["a.ckpt,b.ckpt", "fast,slow"])

    assert [job.positional_args for job in plan.jobs] == [
        ["a.ckpt", "fast"],
        ["a.ckpt", "slow"],
        ["b.ckpt", "fast"],
        ["b.ckpt", "slow"],
    ]


def test_mixed_positional_and_hydra_sweep():
    plan = parse_sweep(["ckpt_a,ckpt_b", "lr=1e-3,1e-4", "seed=1"])

    assert len(plan.jobs) == 4
    assert plan.jobs[0].positional_args == ["ckpt_a"]
    assert plan.jobs[0].hydra_overrides == ["lr=1e-3", "seed=1"]
    assert plan.jobs[-1].positional_args == ["ckpt_b"]
    assert plan.jobs[-1].hydra_overrides == ["lr=1e-4", "seed=1"]


def test_bracket_aware_values_are_preserved_in_hydra_overrides():
    plan = parse_sweep(["sizes=[1,2,3]", "name='a,b'"])

    assert len(plan.jobs) == 1
    assert plan.jobs[0].hydra_overrides == ["sizes=[1,2,3]", "name='a,b'"]


def test_variable_params_identified_for_output_names():
    plan = parse_sweep(["data=a,b", "seed=1", "foo,bar"])

    assert plan.jobs[0].variable_params == {"data": "a", "arg2": "foo"}
    assert plan.jobs[-1].variable_params == {"data": "b", "arg2": "bar"}


def test_allowed_control_parameters_are_swept_before_control_detection():
    plan = parse_sweep(["GPU=2,4", "PARTITION=a,b", "BATCH=64,128", "lr=1e-3"])

    assert len(plan.jobs) == 8
    assert plan.jobs[0].tokens == ["GPU=2", "PARTITION=a", "BATCH=64", "lr=1e-3"]
    assert plan.jobs[-1].tokens == ["GPU=4", "PARTITION=b", "BATCH=128", "lr=1e-3"]
    assert plan.jobs[-1].variable_params == {"GPU": "4", "PARTITION": "b", "BATCH": "128"}


def test_other_control_parameter_sweeps_require_confirmation():
    with pytest.raises(SweepConfirmationRequired) as exc_info:
        parse_sweep(["TIME=1,2", "GPU=2,4"])

    assert exc_info.value.control_names == ["TIME"]


def test_confirmed_other_control_parameter_sweeps_are_allowed():
    plan = parse_sweep(["TIME=1,2"], confirm_control_sweeps=True)

    assert [job.tokens for job in plan.jobs] == [["TIME=1"], ["TIME=2"]]

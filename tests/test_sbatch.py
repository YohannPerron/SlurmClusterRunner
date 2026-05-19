from dataclasses import replace

from src.args import ControlParams
from src.models import EnvironmentConfig, PartitionConfig, PathsConfig, ResourcesConfig, SlurmConfig
from src.paths import plan_run_paths
from src.resources import calculate_resources
from src.sbatch import SbatchContext, render_sbatch
from src.sweep import parse_sweep
from src.time import resolve_time


def partition() -> PartitionConfig:
    return PartitionConfig(
        name="p",
        remote_host="host",
        gpu_type="a100",
        slurm=SlurmConfig(
            account="acc",
            partition="gpu",
            qos="normal",
            dev_qos="dev",
            max_time_hours=10,
            job_name_prefix="tr",
            gres="gpu",
            constraint="a100",
            default_time="02:00:00",
            hint="nomultithread",
            signal="SIGUSR1@600",
        ),
        resources=ResourcesConfig(
            gpu_per_node=8,
            cpu_per_gpu=4,
            task_mode="per_gpu",
            exclusive=True,
            multi_gpu_overrides={"trainer.strategy": "ddp"},
        ),
        paths=PathsConfig(remote_launcher_dir="/launcher", default_project_dir="/proj", data_dir="/data", log_dir="/logs"),
        environment=EnvironmentConfig(
            activate_command="env activate custom",
            shell_init="/home/me/.bashrc",
            exports={"FOO": "bar baz"},
            pre_run=["echo ready"],
            wandb={"set_name": True, "set_group": True},
        ),
        modules={"purge": True, "load": ["cuda"]},
        default_overrides={"num_workers": "6"},
        launcher={"command_prefix": "srun"},
    )


def context(controls: ControlParams, tokens: list[str] | None = None) -> SbatchContext:
    p = partition()
    job = parse_sweep(tokens or ["data.csv", "lr=1e-3"]).jobs[0]
    resources = calculate_resources(controls.gpu, p)
    paths = plan_run_paths(p, "train.py", controls, job, index=0, timestamp="ts")
    return SbatchContext(p, "/proj", "train.py", controls, job, resources, resolve_time(controls, p), paths)


def test_header_contains_resource_directives_and_tag() -> None:
    script = render_sbatch(context(ControlParams(gpu=2, tag="ticket-1")))

    assert "#SBATCH --account=acc" in script
    assert "#SBATCH --qos=normal" in script
    assert "#SBATCH --gres=gpu:2" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gpus-per-node=2" in script
    assert "#SBATCH --ntasks-per-node=2" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --time=02:00:00" in script
    assert "#SBATCH --comment=ticket-1" in script
    assert "#SBATCH --exclusive" in script


def test_environment_and_command_are_rendered() -> None:
    script = render_sbatch(context(ControlParams(gpu=1)))

    assert script.startswith("#!/bin/bash -l\n")
    assert "module purge" in script
    assert "module load cuda" in script
    assert "source /home/me/.bashrc" in script
    assert script.index("source /home/me/.bashrc") < script.index("module purge")
    assert script.index("source /home/me/.bashrc") < script.index("module load cuda")
    assert "env activate custom" in script
    assert "export FOO='bar baz'" in script
    assert "echo ready" in script
    assert "srun python -u train.py data.csv lr=1e-3" in script
    assert "hydra.run.dir=/logs/train/ts/0" in script


def test_activate_command_is_used_for_activation() -> None:
    base = partition()
    p = replace(base, environment=replace(base.environment, activate_command="micromamba activate custom"))
    controls = ControlParams(gpu=1)
    job = parse_sweep(["lr=1e-3"]).jobs[0]
    resources = calculate_resources(controls.gpu, p)
    paths = plan_run_paths(p, "train.py", controls, job, index=0, timestamp="ts")
    ctx = SbatchContext(p, "/proj", "train.py", controls, job, resources, resolve_time(controls, p), paths)

    script = render_sbatch(ctx)

    assert "micromamba activate custom" in script
    assert "env activate" not in script


def test_activate_command_is_used_when_configured() -> None:
    base = partition()
    p = replace(base, environment=replace(base.environment, activate_command="env activate custom"))
    controls = ControlParams(gpu=1)
    job = parse_sweep(["lr=1e-3"]).jobs[0]
    resources = calculate_resources(controls.gpu, p)
    paths = plan_run_paths(p, "train.py", controls, job, index=0, timestamp="ts")
    ctx = SbatchContext(p, "/proj", "train.py", controls, job, resources, resolve_time(controls, p), paths)

    script = render_sbatch(ctx)

    assert "env activate custom" in script


def test_null_activate_command_skips_activation() -> None:
    base = partition()
    p = replace(base, environment=replace(base.environment, activate_command=None))
    controls = ControlParams(gpu=1)
    job = parse_sweep(["lr=1e-3"]).jobs[0]
    resources = calculate_resources(controls.gpu, p)
    paths = plan_run_paths(p, "train.py", controls, job, index=0, timestamp="ts")
    ctx = SbatchContext(p, "/proj", "train.py", controls, job, resources, resolve_time(controls, p), paths)

    script = render_sbatch(ctx)

    assert "activate custom" not in script


def test_batch_multi_gpu_and_wandb_overrides() -> None:
    script = render_sbatch(context(ControlParams(gpu=4, batch=128), ["lr=1e-3,1e-4"]))

    assert "data.batch_size=32" in script
    assert "trainer.strategy=ddp" in script
    assert "trainer.devices=4" in script
    assert "trainer.num_nodes=1" in script
    assert "wandb.name=0_lr=1e-3" in script
    assert "wandb.group=/logs/train/ts-lr" in script


def test_dev_uses_dev_qos() -> None:
    script = render_sbatch(context(ControlParams(dev=True)))
    assert "#SBATCH --qos=dev" in script


def test_time_min_is_omitted_when_not_lower_than_job_time() -> None:
    base = partition()
    p = replace(
        base,
        slurm=replace(
            base.slurm,
            min_time="19:00:00",
            use_min_time_default=True,
            dev_time="01:00:00",
        ),
    )
    job = parse_sweep(["lr=1e-3"]).jobs[0]
    controls = ControlParams(dev=True)
    resources = calculate_resources(controls.gpu, p)
    paths = plan_run_paths(p, "train.py", controls, job, index=0, timestamp="ts")
    ctx = SbatchContext(p, "/proj", "train.py", controls, job, resources, resolve_time(controls, p), paths)

    script = render_sbatch(ctx)

    assert "#SBATCH --time=01:00:00" in script
    assert "#SBATCH --time-min" not in script


def test_jz_h100_script_matches_expected_shape_with_obfuscated_account() -> None:
    p = PartitionConfig(
        name="jz-h100",
        remote_host="JZ",
        gpu_type="H100",
        slurm=SlurmConfig(
            account="redacted@h100",
            partition=None,
            qos="qos_gpu_h100-t3",
            dev_qos="qos_gpu_h100-dev",
            max_time_hours=20,
            job_name_prefix="tr",
            gres="gpu",
            constraint="h100",
            default_time="19:59:59",
            min_time="19:00:00",
            use_min_time_default=True,
            hint="nomultithread",
            signal="SIGUSR1@600",
            output="/lustre/fsn1/projects/rech/msf/uih12tb/logs/outputs/test%j/log.txt",
            error="/lustre/fsn1/projects/rech/msf/uih12tb/logs/outputs/test%j/log.txt",
        ),
        resources=ResourcesConfig(
            gpu_per_node=4,
            cpu_per_gpu=24,
            task_mode="per_gpu",
            multi_gpu_overrides={"trainer": "ddp"},
        ),
        paths=PathsConfig(
            remote_launcher_dir="work/Utils",
            default_project_dir="/lustre/fswork/projects/rech/ufh/ult23zz/AnySat",
            data_dir="/lustre/fsn1/projects/rech/ufh/ult23zz/datasets/",
            log_dir="/lustre/fsn1/projects/rech/msf/uih12tb/logs_AnySat/",
        ),
        environment=EnvironmentConfig(
            activate_command="env activate custom",
            shell_init="/linkhome/rech/genuvt01/uih12tb/.bashrc",
            exports={
                "SRUN_CPUS_PER_TASK": "$SLURM_CPUS_PER_TASK",
                "PYTHONPATH": "./src/utils/",
                "TMPDIR": "/lustre/fsn1/projects/rech/msf/uih12tb/tmp",
                "WANDB__SERVICE_WAIT": "300",
                "WANDB_MODE": "offline",
            },
        ),
        modules={"purge": True, "load": ["arch/h100", "git"]},
        default_overrides={
            "num_workers": "8",
            "hydra/job_logging": "default",
            "hydra/hydra_logging": "default",
            "paths.data_dir": "/lustre/fsn1/projects/rech/ufh/ult23zz/datasets/",
            "paths.log_dir": "/lustre/fsn1/projects/rech/msf/uih12tb/logs_AnySat/",
            "offline": "True",
        },
        launcher={"command_prefix": "srun"},
    )
    controls = ControlParams(name="Ub_Maybe_Final", gpu=8)
    job = parse_sweep([
        "exp=GeoPlex_Uni_MAE",
        "model.network.instance.predictor.dates_to_select=4",
        "seed=43",
    ]).jobs[0]
    resources = calculate_resources(controls.gpu, p)
    paths = plan_run_paths(p, "src/train.py", controls, job, index=4, timestamp="20260402-1558")
    ctx = SbatchContext(
        p,
        "/lustre/fswork/projects/rech/msf/uih12tb/AnySat",
        "src/train.py",
        controls,
        job,
        resources,
        resolve_time(controls, p),
        paths,
    )

    script = render_sbatch(ctx)

    assert "#SBATCH --account=redacted@h100" in script
    assert "#SBATCH --job-name=tr-train-Ub_Maybe_Final" in script
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --gres=gpu:4" in script
    assert "#SBATCH --cpus-per-task=24" in script
    assert "#SBATCH --hint=nomultithread" in script
    assert "#SBATCH --ntasks-per-node=4" in script
    assert "#SBATCH --time=19:59:59" in script
    assert "#SBATCH --time-min=19:00:00" in script
    assert "#SBATCH --qos=qos_gpu_h100-t3" in script
    assert "#SBATCH --output=/lustre/fsn1/projects/rech/msf/uih12tb/logs/outputs/test%j/log.txt" in script
    assert "#SBATCH --error=/lustre/fsn1/projects/rech/msf/uih12tb/logs/outputs/test%j/log.txt" in script
    assert "#SBATCH --constraint=h100" in script
    assert "#SBATCH --signal=SIGUSR1@600" in script
    assert "module purge" in script
    assert "module load arch/h100" in script
    assert "module load git" in script
    assert "env activate custom" in script
    assert "cd /lustre/fswork/projects/rech/msf/uih12tb/AnySat" in script
    assert "srun python -u src/train.py" in script
    assert "trainer=ddp" in script
    assert "trainer.devices=4" in script
    assert "trainer.num_nodes=2" in script
    assert "num_workers=8" in script
    assert "hydra.run.dir=/lustre/fsn1/projects/rech/msf/uih12tb/logs_AnySat/train/20260402-1558-Ub_Maybe_Final/4" in script

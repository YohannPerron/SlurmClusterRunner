# SlurmClusterRunner

SlurmClusterRunner is a Python CLI for launching SLURM jobs from a local machine. It reads cluster/partition definitions from YAML, expands Hydra-style and positional parameter sweeps, renders one `run.sbatch` file per expanded job, submits each job over SSH, and writes the submitted SLURM job id to `job_id.txt` in the remote run directory.

## Installation

This project uses `uv`.

For development from the repository:

```bash
uv sync
uv run runner --help
```

To install SlurmClusterRunner as a normal shell command (`runner`) from this checkout:

```bash
uv tool install -e .
```

Make sure the uv tool directory is on your `PATH`:

```bash
uv tool update-shell
exec "$SHELL"
```

Then call it directly from bash/zsh/fish/etc.:

```bash
runner --help
runner <executable> [ARG_OR_KEY=VALUE ...]
runner -v <executable> [ARG_OR_KEY=VALUE ...]
```

To upgrade after pulling changes:

```bash
uv tool install -e . --reinstall
```

To uninstall:

```bash
uv tool uninstall slurmclusterrunner
```

## Basic usage

```bash
uv run runner <executable> [ARG_OR_KEY=VALUE ...]
```

Add `-v`/`--verbose` to print external `ssh`, `rsync`, `git`, and `sbatch` commands with timing information while debugging slow submissions.

- `<executable>` is the Python script to run from inside the selected partition's configured `paths.default_project_dir`.
- Remaining tokens are either forwarded positional arguments or Hydra-style `key=value` overrides.

Example:

```bash
uv run runner train.py \
  NAME=baseline \
  PARTITION=jz-a100 \
  GPU=4 \
  BATCH=128 \
  TAG=first-run \
  model=resnet,vit \
  lr=1e-3,1e-4
```

This submits four jobs: `2` models times `2` learning rates. Each generated command resembles:

```bash
srun python -u train.py model=resnet lr=1e-3 data.batch_size=32 trainer=ddp ...
```

## Control parameters

Control parameters are consumed by SlurmClusterRunner and are not forwarded directly to your script.

| Parameter | Default | Description |
| --- | --- | --- |
| `NAME` | none | Human-readable run label used in SLURM job names and log directory names. |
| `PARTITION` | configured default | Selects a YAML file from `partitions/` by its `name`. |
| `GPU` | `1` | Total GPUs requested for one job. Multi-node requests must be divisible by `resources.gpu_per_node`. |
| `CPU` | partition default | CPU cores requested for a CPU-only job. Using it with a GPU partition is an error. |
| `BATCH` | none | Total batch size. SlurmClusterRunner injects `data.batch_size=BATCH/GPU`; `BATCH` must be divisible by `GPU`. |
| `TIME` | partition default/max | Wall time as `HH:MM:SS` or integer hours. |
| `MINTIME` | partition default | Boolean controlling whether `#SBATCH --time-min` is emitted when configured. |
| `DEV` | `false` | Boolean. Uses `slurm.dev_qos` and `slurm.dev_time` when available. |
| `CONDA_ENV` | partition environment | Overrides the configured conda environment. |
| `TAG` | none | Stored as `#SBATCH --comment`; required when the partition sets `slurm.require_tag: true`. |
| `GPU_TYPE` | none | Deprecated compatibility alias for `PARTITION`; prefer `PARTITION`. |

Boolean values accept forms like `true/false`, `yes/no`, `1/0`, and `on/off`.

Sweeping `GPU`, `CPU`, `PARTITION`, and `BATCH` is allowed by default. Sweeping other control parameters requires:

```bash
uv run runner --allow-control-sweeps <executable> NAME=a,b ...
```

## Parameter sweeps

Any comma-separated value creates a sweep axis. SlurmClusterRunner expands all axes as a Cartesian product.

```bash
uv run runner train.py PARTITION=jz-a100 seed=1,2 optimizer=adam,sgd
```

Bracketed or quoted commas are preserved:

```bash
uv run runner train.py 'model.layers=[64,128,256]' 'name="a,b"'
```

Positional arguments can also be swept and keep their original order:

```bash
uv run runner scripts/eval.py \
  checkpoint_a.ckpt,checkpoint_b.ckpt \
  val \
  seed=1,2
```

This expands to:

```bash
python -u scripts/eval.py checkpoint_a.ckpt val seed=1
python -u scripts/eval.py checkpoint_a.ckpt val seed=2
python -u scripts/eval.py checkpoint_b.ckpt val seed=1
python -u scripts/eval.py checkpoint_b.ckpt val seed=2
```

## What SlurmClusterRunner creates remotely

For every expanded job, SlurmClusterRunner:

1. Verifies/synchronizes the local launcher checkout to `paths.remote_launcher_dir` on the selected cluster using `rsync`.
2. Creates a run directory under `paths.log_dir`.
3. Writes `run.sbatch` in that directory.
4. Runs `sbatch run.sbatch` over SSH.
5. Writes the parsed job id to `job_id.txt`.

Run directories are named like:

```text
<log_dir>/<executable_stem>[_DEV]/<timestamp>[-NAME][-var_names]/<index>_<values>/
```

The CLI summary prints the selected partition, number of submitted jobs, remote host, run root, and job ids.

## Configuring a new cluster or partition

Cluster behavior is configured entirely through YAML files in `partitions/`. Add one file per runnable partition/project combination, for example `partitions/my-cluster-a100.yaml`.

Use this template as a starting point:

```yaml
name: my-a100
remote_host: my-login-host
gpu_type: A100
default: false

slurm:
  account: my_account        # use null if not required
  partition: gpu             # use null if not required
  qos: normal_gpu            # use null if not required
  dev_qos: dev_gpu           # use null if not supported
  max_time_hours: 20
  default_time: '19:59:59'
  dev_time: '01:59:59'
  min_time: '19:00:00'
  use_min_time_default: true
  require_tag: false
  job_name_prefix: tr
  gres: gpu
  constraint: a100
  hint: nomultithread
  signal: SIGUSR1@600
  output: /remote/logs/outputs/run%j.txt
  error: /remote/logs/outputs/run%j.txt

resources:
  gpu_per_node: 8
  cpu_per_gpu: 8
  task_mode: per_gpu         # per_gpu or per_node
  require_full_node_gpu: false
  exclusive: false
  multi_gpu_overrides:
    trainer: ddp

modules:
  purge: true
  load:
    - cuda/12.1
    - git

environment:
  shell_init: /home/user/.bashrc
  conda_env: myenv
  exports:
    SRUN_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK
    PYTHONPATH: ./src
    WANDB_MODE: offline
  pre_run:
    - echo "Starting job on $(hostname)"
  wandb:
    mode: offline
    set_name: true
    set_group: true
    name_key: logger.wandb.name
    group_key: logger.wandb.group

paths:
  remote_launcher_dir: /remote/work/SlurmClusterRunner
  default_project_dir: /remote/work/MyProject
  data_dir: /remote/data
  log_dir: /remote/logs/MyProject

default_overrides:
  num_workers: '6'

features:
  job_chaining: false

launcher:
  command_prefix: srun

project: MyProject
```

### Required fields

Top-level required fields:

- `name`: value used by `PARTITION=<name>`.
- `remote_host`: SSH host alias or hostname used for `ssh` and `rsync`.
- `gpu_type`: descriptive GPU label.
- `slurm`, `resources`, `paths`, `environment`: required sections.

Required `slurm` fields:

- `account`, `partition`, `qos`, `dev_qos`: strings or `null`.
- `max_time_hours`: maximum allowed wall time.
- `require_tag`: whether users must provide `TAG=...`.

Required `resources` fields:

- `gpu_per_node`: GPUs available per node.
- `cpu_per_gpu`: CPU cores to allocate per GPU.
- `task_mode`: `per_gpu` for one task per GPU, or `per_node` for one task per node.

Required `paths` fields:

- `remote_launcher_dir`: where SlurmClusterRunner syncs itself on the cluster.
- `default_project_dir`: default remote project checkout/path.
- `data_dir`: cluster data path for documentation and future use.
- `log_dir`: root directory for run folders.

Required `environment` fields:

- `conda_env`: environment activated before running the script, or `null`.
- `exports`: environment variables exported in the sbatch script; use `{}` if none.

### Resource layout notes

- `task_mode: per_gpu` emits `--ntasks-per-node=<gpus_per_node>` and `--cpus-per-task=<cpu_per_gpu>`.
- `task_mode: per_node` emits one task per node and `--cpus-per-task=<cpu_per_gpu * gpus_per_node>`.
- If `require_full_node_gpu: true`, users must request exactly a full node for single-node jobs.
- For multi-node jobs, `GPU` must be divisible by `gpu_per_node`.
- `multi_gpu_overrides` are injected only when total GPUs are greater than one.

### SLURM header customization

Common keys such as `account`, `partition`, `qos`, `constraint`, `gres`, `nodes`, `gpus-per-node`, `ntasks-per-node`, `cpus-per-task`, `time`, `time-min`, `hint`, `signal`, `output`, `error`, and `comment` are rendered automatically. Extra keys under `slurm` are also emitted as `#SBATCH --<key>=<value>`.

### Selecting defaults

Exactly one YAML file may set `default: true` if you want `PARTITION` to be optional. If several files are default, SlurmClusterRunner fails and asks for an explicit `PARTITION`.

## Development

Run tests with:

```bash
uv run pytest
```

Useful checks:

```bash
uv run ruff check .
uv run mypy .
```

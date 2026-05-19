# Job Launcher — Updated Runner Specification

## Purpose

A CLI tool to emulate Hydra multirun over SLURM. Given a project, an executable,
and a set of parameters, it generates and submits one `.sbatch` file per
parameter combination on a selected SLURM partition.

This document describes the desired updated runner behaviour. It is not a
requirement to modify the current implementation immediately.

---

## Usage

```bash
python runner.py <path_to_project> <executable> [ARG_OR_KEY=val1,val2 ...]
```

### Example

```bash
python runner.py ~/work/AnySat train.py \
    NAME=exp1 \
    PARTITION=jz-a100 \
    GPU=4 \
    TAG=baseline \
    model=resnet,vit \
    lr=1e-3,1e-4
```

This produces 4 jobs: `2 models × 2 learning rates`.

---

## Special / Control Parameters

These parameters are consumed by the launcher and are not forwarded to the
training script unless explicitly stated.

| Parameter   | Type    | Default             | Description |
|-------------|---------|---------------------|-------------|
| `NAME`      | str     | —                   | Job name suffix and output directory label |
| `GPU`       | int     | `1`                 | Total number of GPUs across all nodes |
| `PARTITION` | str     | configured default  | Runtime partition selection. Replaces the old `GPU_TYPE` parameter |
| `BATCH`     | int     | —                   | Total batch size; divided by `GPU` to get per-GPU size, sets `data.batch_size` |
| `TIME`      | str/int | partition max       | Requested wall time (`HH:MM:SS` or integer hours) |
| `MINTIME`   | bool    | partition default   | Whether to add partition-specific minimum time options |
| `DEV`       | bool    | `False`             | Use development/short QoS settings when supported by the partition |
| `CONDA_ENV` | str     | project/partition default | Environment to activate on the remote host |
| `TAG`       | str     | —                   | User-defined tag stored in the SLURM job comment field |

### Renamed Parameter

- `GPU_TYPE` is deprecated and must be replaced by `PARTITION`.
- A compatibility warning may be emitted if `GPU_TYPE` is provided, but new
  configuration and documentation should use `PARTITION` only.

---

## Positional Argument Sweep Expansion

The runner must support grid search over both Hydra-style overrides and normal
positional executable arguments.

Argument classification:

- Tokens of the form `KEY=VALUE` are Hydra-style overrides, unless `KEY` is a
  launcher control parameter.
- Tokens without `=` are normal positional arguments and are forwarded to the
  executable as-is after sweep expansion.
- Comma-separated values in either form create a sweep axis.

Example:

```bash
python runner.py ~/project scripts/eval.py \
    checkpoint_a.ckpt,checkpoint_b.ckpt \
    val \
    seed=1,2
```

This expands over all combinations and executes commands equivalent to:

```bash
python -u scripts/eval.py checkpoint_a.ckpt val seed=1
python -u scripts/eval.py checkpoint_a.ckpt val seed=2
python -u scripts/eval.py checkpoint_b.ckpt val seed=1
python -u scripts/eval.py checkpoint_b.ckpt val seed=2
```

Rules:

- Normal positional arguments participate in sweeps like `KEY=VALUE` arguments.
- Positional argument order is preserved exactly as provided on the CLI.
- Positional arguments are forwarded without artificial names such as `-arg1`.
- Bracket-aware comma parsing should also apply to positional arguments where
  relevant.

---

## Parameter Sweep Behaviour

- Values separated by commas create a sweep axis: `lr=1e-3,1e-4`.
- List-valued arguments, such as `sizes=[1,2,3]`, are preserved as one value via
  bracket-aware comma parsing.
- All combinations are submitted as independent jobs using a Cartesian product.
- Parameters with more than one value are **variable parameters** and are
  reflected in the output directory name.
- Normal positional arguments also participate in sweep expansion.

---

## Partition Configuration

Cluster/partition configuration must be runtime data loaded from YAML files,
not Python modules copied at deploy time.

Each partition is defined by one YAML file, for example:

```yaml
name: jz-a100
gpu_type: A100
remote_host: jean-zay.example.org
default: true

slurm:
  account: my_account
  partition: gpu_p2
  qos: qos_gpu-t3
  dev_qos: qos_gpu-dev
  max_time_hours: 20
  min_time: "19:00:00"
  require_tag: false

resources:
  gpu_per_node: 8
  cpu_per_gpu: 10
  task_mode: per_gpu   # per_gpu or per_node
  multi_gpu_overrides:
    trainer: ddp

project: AnySat

paths:
  remote_launcher_dir: /gpfswork/$USER/job_launcher
  default_project_dir: /gpfswork/$USER/AnySat
  data_dir: /gpfswork/$USER/data
  log_dir: /gpfswork/$USER/logs

environment:
  conda_env: archeao
  exports:
    PYTHONPATH: ./src/utils/
    WANDB_MODE: offline
  wandb:
    mode: offline        # offline, online, disabled, or unset
    set_name: true
    set_group: true

features:
  job_chaining: false
```

Required information per partition:

- Remote host used for submission/synchronization.
- GPU type label (`gpu_type`) used for resource naming/constraints.
- SLURM account name (`slurm.account`, nullable when the cluster does not require one).
- SLURM partition/QoS/dev-QoS settings.
- Maximum wall time and optional minimum time policy.
- Whether `TAG` is required for this partition (`require_tag`).
- `gpu_per_node`.
- `cpu_per_gpu`.
- Task layout mode: `per_gpu` for one task per GPU, or `per_node` for one task
  per node.
- Multi-GPU override values, e.g. `trainer=ddp`.
- A single target project name and its paths. A YAML file should not contain
  path defaults for multiple projects; create another partition/project YAML if
  the same partition is used for another project.
- Default paths and environment settings, including the environment variables
  listed in `environment.exports`.
- WandB policy, including whether to force offline mode, use online mode,
  disable WandB injection, or leave WandB untouched.
- Feature flags, including whether job chaining is enabled.

The runner selects the YAML file using `PARTITION=<name>`. If no partition is
provided, it uses the single partition marked as `default: true`, or fails if the
default is ambiguous.

---

## Remote Version Synchronization

Before submitting jobs or executing launcher commands on a remote host, the
runner must verify that the remote launcher version is up to date with the local
version.

Expected behaviour:

1. Determine the local version, preferably from Git:
   - current commit hash;
   - dirty/uncommitted state if relevant.
2. Query the remote launcher checkout for its current commit hash.
3. If the remote version differs, automatically update the remote copy before
   submission.
4. Fail clearly if synchronization cannot be completed.

Acceptable synchronization strategies include:

- `git fetch` + `git checkout`/`git pull` on the remote checkout;
- `rsync`/`scp` fallback for uncommitted local changes;
- explicit refusal to submit dirty local changes unless a supported sync mode is
  configured.

The invariant is: **the remote runner used for submission must match the local
runner version that initiated the command**.

---

## Multi-GPU / Multi-Node

When `GPU > 1`:

- If `GPU <= gpu_per_node`: request one node with multiple GPUs.
- If `GPU > gpu_per_node`: request multiple nodes.
- `GPU` must be divisible by `gpu_per_node` for multi-node jobs.
- CPU allocation is derived from `cpu_per_gpu` and the selected `task_mode`.
- If `task_mode: per_gpu`, request one task per GPU:
  - `--ntasks-per-node=<gpus_per_node>`
  - `--cpus-per-task=<cpu_per_gpu>`
- If `task_mode: per_node`, request one task per node:
  - `--ntasks-per-node=1`
  - `--cpus-per-task=<cpu_per_gpu × gpus_per_node>`
- Multi-GPU script overrides are read from the selected partition YAML, for
  example:
  - `trainer=ddp`
  - `trainer.devices=<gpus_per_node_for_this_job>`
  - `trainer.num_nodes=<num_nodes>`
  - `trainer.trainer.enable_progress_bar=False`

---

## TAG / SLURM Comment Field

`TAG=<value>` is a launcher control parameter.

- It is not forwarded to the training script.
- It is always saved in the SLURM job comment field, e.g. via
  `#SBATCH --comment=<TAG>` or equivalent submission option.
- It can be used to identify, filter, or group jobs outside of Hydra/WandB.
- Each partition may set `slurm.require_tag: true` to make `TAG` mandatory.
  For now, all generated partition configs set `require_tag: false`.

---

## Job Chaining / Time Splitting

Job chaining is a low-priority feature.

When enabled for a partition and when the requested `TIME` exceeds the
partition's maximum wall time:

- The total duration may be split into `ceil(total / max)` sequential jobs.
- Each job depends on the previous job via `--dependency=afterany:<jobid>`.
- The job name gets an `_{i}/{n}` suffix.

Rules:

- Job chaining must be controlled per partition using a YAML feature flag, e.g.
  `features.job_chaining: true`.
- If chaining is disabled and `TIME` exceeds the maximum, the runner should fail
  validation with a clear error.
- Because this is low priority, the first updated runner may omit chaining as
  long as it validates and reports unsupported requests cleanly.

---

## Output Structure

Run folders are created under the selected partition's `paths.log_dir` unless an
explicit project/log override is provided.

```text
<paths.log_dir>/
  <executable_stem>[_DEV]/
    <YYYYMMDD-HH:MM>[-NAME][-var_params]/
      <i>_[var_param_values]/
        run.sbatch
        job_id.txt
```

For chained jobs, if implemented:

```text
run_0.sbatch
run_1.sbatch
...
job_id.txt
```

After submitting a job, the runner must capture the scheduler job ID returned by
`sbatch` and save it in the run folder next to the generated `.sbatch` file.

Expected behaviour:

- Parse the submitted job ID from the `sbatch` output, e.g.
  `Submitted batch job 123456` → `123456`.
- Write the ID to `job_id.txt` for normal jobs.
- For chained/split jobs, use a single `job_id.txt` file containing one job ID
  per line in chain order.
- If submission succeeds but the job ID cannot be parsed, fail clearly or write
  the raw submission output to a diagnostic file in the run folder.
- The saved job ID files are intended to support later status checks,
  cancellation helpers, and reproducibility.

The `.sbatch` file contains the SLURM header generated from the selected
partition YAML followed by a command of the form:

```bash
python -u <executable> [positional args] [key=value ...] \
    hydra.run.dir='<paths.log_dir>/<run_dir>' \
    [optional WandB name/group overrides]
```

---

## Deprecated Current Cluster Model

The old model used deploy-time Python config files:

- `cluster_JZ.py`
- `cluster_ADA.py`
- `cluster_Star.py`
- `cluster.py`

This model should be removed from the updated runner design. Cluster and
partition data should live in YAML files and be selected at runtime using
`PARTITION`.

---

## WandB Integration

WandB behaviour is configured per partition in the partition YAML. The runner
must not assume that all partitions use offline logging.

Partition-level WandB policy should support at least:

- `mode: offline` → set `WANDB_MODE=offline`.
- `mode: online` → set `WANDB_MODE=online`, or otherwise allow online logging.
- `mode: disabled` → disable WandB injection/export if appropriate.
- `mode: unset` or omitted → do not set `WANDB_MODE`.

When enabled by the partition policy:

- `logger.wandb.name` is set to `<time>_<NAME>_<var-param-values>`.
- `logger.wandb.group` should be stable for a generated sweep, avoiding random
  components that make re-submission difficult to correlate.

Partitions may disable automatic WandB name/group injection entirely via
`environment.wandb.set_name: false` and/or `environment.wandb.set_group: false`.

---

## Known Limitations / Future Work

1. Add a dry-run/preview mode before submission.
2. Add config-file support for job parameters.
3. Add job listing/status tracking using the saved submitted job IDs.
4. Add a cancellation helper for whole sweeps.
5. Implement job chaining if still needed after the partition YAML model is in
   place.

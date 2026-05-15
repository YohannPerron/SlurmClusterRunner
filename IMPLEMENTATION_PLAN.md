# NewRunner Implementation Plan

## 0. Bootstrap the project

Create a proper Python package skeleton:

```text
pyproject.toml
src/newrunner/
  __init__.py
  cli.py
  config.py
  models.py
  args.py
  sweep.py
  resources.py
  time.py
  paths.py
  sbatch.py
  sync.py
  submit.py
  commands.py
tests/
```

Use `uv` for project setup and tooling:

```bash
uv init
uv add pyyaml
uv add --dev pytest ruff mypy
```

---

## 1. Configuration loading

Implement YAML partition loading first.

### Files

- `models.py`
- `config.py`

### Features

- Load all `partitions/*.yaml`.
- Validate required fields:
  - `name`
  - `remote_host`
  - `gpu_type`
  - `slurm.*`
  - `resources.*`
  - `paths.*`
  - `environment.*`
- Select partition by `PARTITION=<name>`.
- If no `PARTITION`, use the single `default: true` partition.
- Fail if no default or multiple defaults.
- Support deprecated `GPU_TYPE` with a compatibility warning and temporary mapping to `PARTITION`.

### Tests

- Valid partition loads.
- Missing required fields fail.
- Ambiguous defaults fail.
- `PARTITION` selection works.
- `GPU_TYPE` emits warning.

---

## 2. CLI and control parameter parsing

### Files

- `cli.py`
- `args.py`

### CLI shape

```bash
python -m newrunner <path_to_project> <executable> [ARG_OR_KEY=val1,val2 ...]
```

Later expose a script entry point:

```bash
newrunner <path_to_project> <executable> ...
```

### Control parameters

Parse and consume:

```text
NAME
GPU
PARTITION
BATCH
TIME
MINTIME
DEV
CONDA_ENV
TAG
```

Do not forward these to the training script.

### Type validation

- `GPU`: int, default `1`
- `BATCH`: optional int
- `TIME`: `HH:MM:SS` or integer hours
- `MINTIME`: bool
- `DEV`: bool
- `TAG`: optional str
- `CONDA_ENV`: optional str

### Tests

- Control params removed from forwarded args.
- Defaults applied.
- Invalid types fail clearly.
- `TAG` required when partition config has `slurm.require_tag: true`.

---

## 3. Bracket-aware sweep parsing

### Files

- `sweep.py`
- `args.py`

Implement bracket-aware comma splitting.

Examples:

```text
lr=1e-3,1e-4 -> ["1e-3", "1e-4"]
sizes=[1,2,3] -> ["[1,2,3]"]
checkpoint_a.ckpt,checkpoint_b.ckpt -> ["checkpoint_a.ckpt", "checkpoint_b.ckpt"]
```

Classify tokens:

- `KEY=VALUE` and not a control param -> Hydra override.
- no `=` -> positional argument.
- both can produce sweep axes.

Generate Cartesian products while preserving positional order.

### Tests

- Hydra-only sweep.
- Positional-only sweep.
- Mixed positional and Hydra sweep.
- Bracket-aware values are preserved.
- Variable params are identified for output names.

---

## 4. Resource calculation

### Files

- `resources.py`
- `time.py`

Given `GPU`, partition config, and `task_mode`, compute:

```python
nodes
gpus_per_node
ntasks_per_node
cpus_per_task
total_gpus
```

Rules:

- If `GPU <= gpu_per_node`: request one node.
- If `GPU > gpu_per_node`: require divisibility by `gpu_per_node`.
- `task_mode: per_gpu`:
  - `ntasks_per_node = gpus_per_node`
  - `cpus_per_task = cpu_per_gpu`
- `task_mode: per_node`:
  - `ntasks_per_node = 1`
  - `cpus_per_task = cpu_per_gpu * gpus_per_node`

Also support existing YAML fields like:

```yaml
resources:
  require_full_node_gpu: true
  exclusive: true
```

### Time validation

- Use partition default time if absent.
- Use dev time when `DEV=true` if configured.
- Fail if requested time exceeds max and chaining is disabled.
- Initially omit chaining, but validate clearly.

### Tests

- Single GPU.
- Multi-GPU one node.
- Multi-node valid.
- Multi-node invalid divisibility.
- `per_gpu` vs `per_node`.
- Time max validation.

---

## 5. Output/run directory planning

### Files

- `paths.py`

Generate remote run folders like:

```text
<log_dir>/<task_name>/<YYYYMMDD-HH:MM>[-NAME][-var_params]/<i>_[var_param_values]/
```

Where:

- `task_name` can come from executable stem, e.g. `train.py` -> `train`.
- timestamp is generated once per invocation.
- variable params are sweep axes with more than one value.
- path components are sanitized.

### Tests

- Stable timestamp reused across all jobs in one sweep.
- Variable params included.
- Positional sweep values reflected safely.
- Names sanitized.

---

## 6. sbatch script generation

### Files

- `sbatch.py`

Generate one script per expanded combination.

### SLURM header

From YAML:

- account
- job name
- partition
- qos/dev_qos
- constraint
- gres
- nodes
- gpus per node
- ntasks per node
- cpus per task
- time
- hint
- signal
- output/error
- comment from `TAG`
- exclusive if configured

### Environment setup

From YAML:

- module purge/load
- source shell init
- conda activate
- exports
- pre-run commands

### Command

Form:

```bash
python -u <executable> [positional args] [key=value ...] hydra.run.dir='<run_dir>'
```

Apply additional injected overrides:

- `BATCH` -> `data.batch_size=<BATCH/GPU>`
- multi-GPU overrides from YAML when `GPU > 1`
- `trainer.devices=<gpus_per_node>`
- `trainer.num_nodes=<nodes>`
- partition `default_overrides`
- WandB name/group if enabled

Respect:

```yaml
launcher:
  command_prefix: srun
```

So final command may be:

```bash
srun python -u train.py ...
```

### Tests

- Header contains correct SLURM directives.
- Control params are not forwarded.
- Batch divided by GPU.
- Multi-GPU overrides injected.
- WandB policy respected.
- `TAG` becomes `#SBATCH --comment=...`.

---

## 7. Submission and job ID capture

### Files

- `submit.py`
- `commands.py`

Introduce a command abstraction so tests can mock subprocess calls.

Submission flow:

1. Create remote run directory.
2. Copy or write `run.sbatch`.
3. Run `sbatch run.sbatch` remotely.
4. Parse output, e.g. `Submitted batch job 123456`.
5. Write `job_id.txt` in the run folder.
6. If parsing fails, save raw output to a diagnostic file and fail clearly.

### Tests

Mock `ssh`, `scp`/`rsync`, and `sbatch`.

- Successful submission writes job ID.
- Bad sbatch output fails clearly.
- Remote mkdir/write commands called correctly.
- No real remote commands during tests.

---

## 8. Remote version synchronization

### Files

- `sync.py`

Implement before submission.

Initial safe strategy:

1. Get local commit:

```bash
git rev-parse HEAD
git status --porcelain
```

2. If dirty:
   - initially refuse submission with a clear message.
   - later add rsync mode if needed.
3. Query remote:

```bash
ssh <host> 'cd <remote_launcher_dir> && git rev-parse HEAD'
```

4. If mismatch:

```bash
ssh <host> 'cd <remote_launcher_dir> && git fetch && git checkout <local_hash>'
```

5. Fail if remote cannot be synced.

### Tests

- Matching versions do nothing.
- Mismatch triggers update.
- Dirty local checkout refuses.
- Remote failures propagate useful error.

---

## 9. CLI orchestration

### File

- `cli.py`

Pipeline:

```text
parse CLI
extract control params
load partition config
validate controls
sync remote launcher
expand sweep
compute resources
plan output dirs
generate sbatch scripts
submit each job
write job_id.txt
print summary
```

Useful summary:

```text
Partition: jz-a100
Jobs: 4
GPU/job: 4
Remote host: JZ
Run root: ...
Submitted:
  0_model=resnet_lr=1e-3 -> 123456
  ...
```

---

## 10. Job chaining

Defer implementation initially.

Required first behavior:

- If `TIME` exceeds partition max and `features.job_chaining: false`, fail.
- If `TIME` exceeds max and `features.job_chaining: true`, either:
  - implement chaining, or
  - fail with `job chaining configured but not yet implemented`.

Later implementation:

- Split total duration into chunks.
- Generate `run_0.sbatch`, `run_1.sbatch`, ...
- Submit with dependency:

```bash
sbatch --dependency=afterany:<previous_job_id> run_i.sbatch
```

- Write all IDs to one `job_id.txt`.

---

## 11. Test plan

Use `pytest`.

Prioritize pure unit tests before remote behavior.

Recommended test files:

```text
tests/test_config.py
tests/test_args.py
tests/test_sweep.py
tests/test_resources.py
tests/test_time.py
tests/test_paths.py
tests/test_sbatch.py
tests/test_submit.py
tests/test_sync.py
tests/test_cli.py
```

Important coverage:

- YAML validation.
- Partition selection.
- Control parameter parsing.
- Hydra and positional sweep expansion.
- Bracket-aware comma parsing.
- Multi-GPU resource math.
- `task_mode: per_gpu`.
- `task_mode: per_node`.
- sbatch generation.
- WandB injection.
- job ID parsing/writing.
- remote sync decision logic.

---

## Suggested implementation order

1. Project skeleton and dependencies.
2. Config models and YAML loading.
3. CLI/control parsing.
4. Sweep expansion.
5. Resource/time validation.
6. sbatch generation.
7. Path/run directory planning.
8. Submission abstraction and job ID parsing.
9. Remote version sync.
10. End-to-end CLI tests.
11. Optional job chaining.

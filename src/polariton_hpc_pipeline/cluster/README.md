# Cluster job wrappers — ICM UW (Rysy)

## Files

| File | Purpose |
|------|---------|
| `job_gpu.sh` | Wrapper for GPU stages. Loads CUDA modules and runs the GPU-stage command. When `ICM_RYSY_NVME=1` is exported by `submit.sh`, it sets `SCRATCH=/scratch/<slurm_job_id>`. |
| `job_stage.sh` | Wrapper for non-GPU stages on Rysy. No CUDA modules; used for the lightweight `finalize` stage. |

## Execution flow

```text
rysy login node
  -> submit.sh
  -> optionally sbatch legacy threshold search
  -> sbatch scenario array with afterok dependency
       each task: simulate on NVMe scratch -> render inline -> copy back artifacts
  -> sbatch finalize with afterok dependency
```

There is no Topola stage in the supported flow. `submit.sh` is the only supported
entrypoint. It exits after submitting the dependent Slurm jobs unless `--wait`
is passed.

## Scratch on Rysy

NVMe-oF scratch is requested only for scenario jobs that produce heavy transient
HDF5 output.

| Stage | Caller | Resources | `ICM_RYSY_NVME` | Scratch usage |
|-------|--------|-----------|-----------------|---------------|
| `threshold_search.py` (legacy non-sweep mode only) | `_rysy_sbatch` | `gpu:N` | not set | no |
| `run_scenario.py` | `_rysy_sbatch_nvme` | `gpu:N,nvme:SIZE` | `1` | yes |
| `finalize.py` | `_rysy_sbatch` | CPU only | not set | no |

`job_gpu.sh` sets `SCRATCH=/scratch/<slurm_job_id>` only when `ICM_RYSY_NVME=1`.
It accepts both `SLURM_JOB_ID` and `SLURM_JOBID` and exits with an error if
neither is set. The scratch path is deleted when the job exits.

`job_gpu.sh` does not request resources itself; `submit.sh` does that via
`sbatch`.

Do not set `SCRATCH` in `slurm.env`. Use `NVME_GB` to control the scratch size.

## Storage layout

| Path | Visibility | Lifecycle |
|------|------------|-----------|
| `/scratch/<slurm_job_id>` | Rysy compute node only | Deleted on job exit |
| `/lu/tetyda/home/$USER/...` | Login node + Rysy | Persistent run directory and copied-back artifacts |

## Required `slurm.env`

Create the repository-root file from `slurm.env.example`. The variables below
are required by `polariton_hpc_pipeline`; variables for the other submission
wrappers may remain in the same shared file.

```bash
SLURM_ACCOUNT=g100-2262
SLURM_PARTITION=gpu
SLURM_QOS=normal
SLURM_TIME=04:00:00
SLURM_MEM=40GB
SLURM_GPUS=1
SLURM_CPUS=4
NVME_GB=100

TETYDA_RUNS_BASE=/lu/tetyda/home/MYUSER/polariton/runs

MAX_CONCURRENT_SCENARIOS=2
FINALIZE_MEM=8G
FINALIZE_CPUS=4
FINALIZE_TIME=00:30:00
TIME_RESPONSE_MEM=4G
TIME_RESPONSE_CPUS=4
TIME_RESPONSE_TIME=00:30:00
```

`submit.sh` exports the current repository checkout as `PROJECT_ROOT` to every
submitted job. Run it from the checkout you want the jobs to use, ideally on
`/lu/tetyda`.

If your NVENC-capable `ffmpeg` is not on `PATH`, export `FFMPEG_BIN` in the
shell before invoking `submit.sh`.

## Modules

Module names (`common/python/3.13.2`, `gpu/cuda/12.1`, etc.) are specific to
Rysy as of the current setup. Verify with `module avail` if jobs fail during
module loading.

## Where to run it

Run `submit.sh` directly on a Rysy login node. It no longer performs an
automatic SSH hop from `hpc.icm.edu.pl`.

# Cluster job wrappers — Rysy at ICM UW

## Files

| File | Purpose |
|------|---------|
| `job_gpu.sh` | Wrapper for GPU stages (threshold search, scenario simulation). Loads CUDA modules. Always requests `--gres=gpu:N`. |
| `job_stage.sh` | Wrapper for non-GPU stages (visualization, finalize). Does **not** load CUDA modules. May still run on GPU-equipped nodes — see resource model below. |

## Resource model for Rysy

Rysy is a **GPU cluster** at ICM UW.  All public Slurm documentation for
Rysy uses `--gres=gpu:...`.  There is no confirmed dedicated CPU-only
partition available for this workflow.

### What this means in practice

Non-GPU stages (visualization, finalize) submitted via `job_stage.sh` will:

1. **Run on the same GPU partition** as GPU jobs if `CPU_PARTITION` is not
   set or is identical to `SLURM_PARTITION` in `slurm.env`.
   `submit.sh` prints a **WARNING** when this is the case.

2. **Run on a separate partition** if `CPU_PARTITION` is set in `slurm.env`
   to a value different from `SLURM_PARTITION`.  In this case no warning is
   shown.

These jobs do not allocate a GPU (`--gres=gpu:...` is not passed), but they
occupy a slot on a GPU node.  This is acceptable for short/light postprocessing
but is not equivalent to a true CPU-only queue.

### Configuring a separate partition

Add to `slurm.env`:

```bash
CPU_PARTITION=cpu          # or whatever Rysy makes available
CPU_MEM=16G
CPU_CPUS=8
CPU_TIME=01:00:00
```

Leave these unset to fall back to the GPU partition (with the warning).

## Module versions

The module names (`common/python/3.13.2`, `gpu/cuda/12.1`, etc.) are
specific to Rysy as of the initial pipeline setup.  Verify with
`module avail` if jobs fail at the module load step.

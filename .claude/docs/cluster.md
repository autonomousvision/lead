# Cluster Information

| Cluster | Partition | Max time | GPUs |
|---------|-----------|----------|------|
| TCML | `L40Sday` | `7-00:00:00` | max 4 (`--gres=gpu:4`) |
| MLCloud | `a100-galvani` | `3-00:00:00` | max 4 (`--gres=gpu:4`) |
| Bare metal | *(no SLURM)* | unlimited | use free GPUs |

- Each job should request **at most 4 GPUs**.
- Bare metal jobs run directly without SLURM — the `train` function in `slurm/init.sh` detects this automatically when `sbatch` is not available.

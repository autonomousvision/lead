#!/bin/bash
#SBATCH --job-name="vs"
#SBATCH --partition=L40Sday
#SBATCH --time=10:00:00
#SBATCH --gpus=1
#SBATCH --output=outputs/slurm_logs/vscode_%j.out
#SBATCH --error=outputs/slurm_logs/vscode_%j.out


/usr/sbin/sshd -D -p 8732 -f /dev/null -h ${HOME}/.ssh/id_rsa # uses the user key as the host key

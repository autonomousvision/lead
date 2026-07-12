#!/bin/bash
#SBATCH --job-name="vs"
#SBATCH --partition=a100-galvani
#SBATCH --time=10:00:00
#SBATCH --gpus=1

/usr/sbin/sshd -D -p 8732 -f /dev/null -h ${HOME}/.ssh/id_ecdsa # uses the user key as the host key

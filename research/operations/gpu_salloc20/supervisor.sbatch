#!/bin/bash
#SBATCH --job-name=gpu-pool20-supervisor
#SBATCH --partition=defq
#SBATCH --account=hpcusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=45-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@3600
#SBATCH --chdir=/opt/craniofacial/operations/gpu_salloc20
#SBATCH --output=/opt/craniofacial/operations/gpu_salloc20/logs/supervisor-%j.log
set -uo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
pool_dir=/opt/craniofacial/operations/gpu_salloc20
# Queue a successor at the controller's own time limit. GPU allocations are independent.
trap '/opt/site/slurm/bin/sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" "$pool_dir/supervisor.sbatch" >> "$pool_dir/logs/supervisor-successors.log" 2>&1' USR1
while true; do
    /usr/bin/python -B "$pool_dir/pool.py" --standby &
    pool_guardian_pid=$!
    while kill -0 "$pool_guardian_pid" 2>/dev/null; do
        wait "$pool_guardian_pid" || true
    done
    sleep 10
done

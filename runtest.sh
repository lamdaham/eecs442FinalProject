#!/bin/bash
# The interpreter used to execute the script
# "#SBATCH" directives that convey submission options:
SBATCH --job-name=example_job
SBATCH --mail-user=ivanlam@umich.edu
SBATCH --mail-type=BEGIN,END
SBATCH --cpus-per-task=8
SBATCH --nodes=1
SBATCH --ntasks-per-node=1
SBATCH --mem-per-cpu=1000m
SBATCH --time=01:00:00
SBATCH --account=eecs442f25_class
SBATCH --partition=gpu_mig40,gpu,spgpu
SBATCH --output=example_job.log

echo "hello world"
python3 lidarProject.py
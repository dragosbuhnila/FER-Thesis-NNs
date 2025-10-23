#!/bin/bash
#SBATCH --job-name=dbuhnila/tesi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s331394@studenti.polito.it
#SBATCH --partition=global
#SBATCH --time=5:00
#SBATCH --nodes=1
#SBATCH --mem=10000
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0z
#SBATCH --output=err_log/train_%j.log
#SBATCH --error=err_log/train_%j.err
module load intel/python/3/2019.4.088
module load nvidia/cudasdk/11.6
source /home/dbuhnila/myvenv/bin/activate
cd /home/famato/final_scripts/scripts

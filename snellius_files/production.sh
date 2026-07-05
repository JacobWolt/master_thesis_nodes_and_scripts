#!/bin/bash
#SBATCH --job-name=gromacs_production_100ns_npt
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=18:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=m.j.woltjer@umail.leidenuniv.nl

module purge
module load 2025
module load GROMACS/2025.2-foss-2025a

### 100 ns UNRESTRICTED NPT RUN ###

gmx grompp -f npt.mdp -c npt-rotated.gro -r npt-rotated.gro -p topol.top -n index.ndx -o npt.tpr -maxwarn 100
gmx mdrun -v -deffnm npt -ntmpi 24 -ntomp 8 #-cpi nvt_prev.cpt

WORK_DIR=$(pwd)
a=$(basename ${WORK_DIR})
parent=$(basename $(dirname ${WORK_DIR}))
dir="${parent}/${a}"
#
#mkdir -p /gpfs/home1/jwoltjer/MD_simulations/results/equi/${dir}
cp *.gro *.log *.xtc *.edr *.cpt /gpfs/home1/jwoltjer/MD_simulations/results/equi/${dir}/

#!/bin/bash
#SBATCH --job-name=gromacs_1ns_equi_after_100ns_nvt
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=m.j.woltjer@umail.leidenuniv.nl

module purge
module load 2025
module load GROMACS/2025.2-foss-2025a

### ACTUAL RUN COMMAND ###

# export GMX_MAXBACKUP=-1 #gets rit of #nvt.gro1# type files

### 10 ps NVT RUN ###

gmx grompp -f nvt-equi.mdp -c em.gro -r em.gro -p topol.top -n index.ndx -o nvt-equi.tpr -maxwarn 100
gmx mdrun -v -deffnm nvt-equi -ntmpi 24 -ntomp 8 # -cpi nvt_prev.cpt # checkpoint usage

sleep 20

### 40 ps NPT RUN ###

gmx grompp -f npt-equi.mdp -c nvt-equi.gro -r nvt-equi.gro -p topol.top -n index.ndx -o npt-equi.tpr -maxwarn 100
gmx mdrun -v -deffnm npt-equi -ntmpi 24 -ntomp 8 #-cpi nvt_prev.cpt

sleep 20

### 1 ns NPT TEST RUN ###

# gmx grompp -f npt-equi2-tst.mdp -c npt-equi.gro -r npt-equi.gro -p topol.top -n index.ndx -o npt-equi2-tst.tpr -maxwarn 100
# gmx mdrun -v -deffnm npt-equi2-tst -ntmpi 24 -ntomp 8 #-cpi nvt_prev.cpt


### 100 ns NPT RUN ###

gmx grompp -f npt-equi2.mdp -c npt-equi.gro -r npt-equi.gro -p topol.top -n index.ndx -o npt-equi2.tpr -maxwarn 100
gmx mdrun -v -deffnm npt-equi2 -ntmpi 24 -ntomp 8 #-cpi nvt_prev.cpt

sleep 20

gmx grompp -f npt-equi2.mdp -c npt-equi2.gro -r npt-equi2.gro -p topol.top -n index.ndx -o centering.tpr -maxwarn 100
echo -e "1\n0\n" | gmx trjconv -f npt-equi2.gro -s centering.tpr -o centered.gro -center -pbc mol)

WORK_DIR=$(pwd)
a=$(basename ${WORK_DIR})
parent=$(basename $(dirname ${WORK_DIR}))
dir="${parent}/${a}"

mkdir -p /gpfs/home1/jwoltjer/MD_simulations/results/equi/${dir}
cp *.gro *.log *.xtc *.edr *.cpt *.top *.ndx /gpfs/home1/jwoltjer/MD_simulations/results/equi/${dir}

#!/bin/bash
#SBATCH --job-name=gromacs_1ns_equi_after_100ns_nvt
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=m.j.woltjer@umail.leidenuniv.nl


module purge
module load 2025
module load GROMACS/2025.2-foss-2025a

# Go where the job has been launched
cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }

cp *.mdp *.top *.gro *.ndx posre.itp /gpfs/home1/jwoltjer/MD_simulations/run_dir/${SLURM_JOBID}

cd /gpfs/home1/jwoltjer/MD_simulations/run_dir/${SLURM_JOBID}

#Copy forcefield files
cp -r ../Slipids_2020.ff ../SLipids_2020 ../amber19sb.ff ${SLURM_JOBID}

#### Create scratch directory ### (option 1)
#export TEMPWORKDIR=/scratch-shared/${USER}/JOB-${SLURM_JOBID}
#mkdir -p "${TEMPWORKDIR}"
#cp ./* "${TEMPWORKDIR}"
#ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}/JOB-${SLURM_JOBID}"

### Run on personal ### (option 2)
#out previous option
 
# Go to the temporary work directory
#cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }
 
# Run job here
echo "# [$(date)] job ${SLURM_JOB_NAME}"
echo "# [$(date)] Running Gromacs..."

###ACTUAL RUN COMMAND###

export GMX_MAXBACKUP=-1 #gets rit of #nvt.gro1# type files

### Build system ###

gmx editconf -f system_gmx.gro  -o mol_z.gro -rotate 0 0 0 -box 7 7 7 -c

gmx grompp -f em.mdp -c system_gmx.gro  -p system_gmx.top -o em.tpr -maxwarn 10
gmx mdrun -v -deffnm em

gmx insert-molecules -f mol_z.gro -ci POPC.pdb -nmol 128 -o mol_popc.gro  -scale 0.4 -try 1000 -seed 99 -p system_gmx.top

HEADER='# 
; Include forcefield parameters
#include "./amber19sb.ff/forcefield.itp"
#include "./Slipids_2020.ff/forcefield.itp"
'
FOOTER='# 
; Include Position restraint file
#ifdef POSRES
#include "posre.itp"
#endif

; Include water topology
#include "./amber19sb.ff/tip3p.itp"

; Include ions topology
#include "./amber19sb.ff/ions_tip3p.itp"

#ifdef POSRES_WATER
; Position restraint for each water oxygen
[ position_restraints ]
;  i funct       fcx        fcy        fcz
1    1       1000       1000       1000
#endif

#include "./SLipids_2020/itp_files/DPPC.itp"

[ system ]
; Name
Generic title in water

[ molecules ]
; Compound       #mols
system1              1
DPPC                128
'

for file in *.top; do
    # Create temp file with header, original content, and footer
    { echo "$HEADER"; cat "$file"; echo "$FOOTER"; } > "$file.tmp"
    mv "$file.tmp" "$file"
done



### Energy minimization ###

#gmx grompp -f em.mdp -c all.gro  -p system_gmx.top -o em.tpr -maxwarn 10
#gmx mdrun -v -deffnm em

### NVT RUN ###

#gmx grompp -f nvt.mdp -c geometry.gro -r geometry.gro -p topol.top -n index.ndx -o npt.tpr -maxwarn 100
#gmx mdrun -v -deffnm nvt -ntmpi 24 -ntomp 8 # -cpi nvt_prev.cpt # checkpoint usage

#sleep 200

### NPT RUN ###

#gmx grompp -f npt.mdp -c geometry.gro -r geometry.gro -p topol.top -n index.ndx -o npt.tpr -maxwarn 100
#gmx mdrun -v -deffnm npt -ntmpi 24 -ntomp 8 #-cpi nvt_prev.cpt


echo "# [$(date)] Gromacs job finished"
 
## Move back data from the temporary work directory and scratch, and clean-up
#find ./ -type l -delete
#rsync -av "./" "${SLURM_SUBMIT_DIR}/"
# 
#cd "${SLURM_SUBMIT_DIR}" || { echo "Could not go to ${SLURM_SUBMIT_DIR}. Aborting..."; exit 1; }
#
#rmdir "${TEMPWORKDIR}" 2> /dev/null || echo "Leftover files on ${TEMPWORKDIR}"
#[ ! -d "${TEMPWORKDIR}" ] && { [ -h JOB-"${SLURM_JOBID}" ] && rm JOB-"${SLURM_JOBID}"; }

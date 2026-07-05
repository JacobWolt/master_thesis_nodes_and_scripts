#!/bin/bash
#SBATCH --job-name=gromacs_multi_lipid
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=m.j.woltjer@umail.leidenuniv.nl

module purge
module load 2025
module load GROMACS/2025.2-foss-2025a


BEGIN=1
INTERVAL=1 
END=20 

for a in `seq $BEGIN $INTERVAL $END`
do

#  ### Save on personal (option 1) ###
#  
#  mkdir -p /gpfs/home1/jwoltjer/MD_simulations/run_dir/no_metal_double_bonded/$a  
#  
#  cp *.top *.gro *.itp /gpfs/home1/jwoltjer/MD_simulations/run_dir/no_metal_double_bonded/$a
#  
#  cd /gpfs/home1/jwoltjer/MD_simulations/run_dir/no_metal_double_bonded/$a
#
#  #Copy input files
#  cp -r /gpfs/home1/jwoltjer/MD_simulations/base_files/popc/* ./ 
#  
#  #Copy forcefield files
#  cp -r ../../Slipids_2020.ff ../../SLipids_2020 ../../amber19sb.ff ./
  
  ### Save on scratch directory (option 2) ###
  export TEMPWORKDIR=/scratch-shared/${USER}/thesis_md_runs/"$(basename "$SLURM_SUBMIT_DIR")"/$a
  mkdir -p "${TEMPWORKDIR}"
  cp *.top *.gro *.itp "${TEMPWORKDIR}"
#  ln -s "${TEMPWORKDIR}" "${SLURM_SUBMIT_DIR}/JOB-${SLURM_JOBID}"

  # Go to the temporary work directory
  cd "${TEMPWORKDIR}" || { echo "Could not go to ${TEMPWORKDIR}. Aborting..."; exit 1; }
  
  #Copy input files
  cp -r /gpfs/home1/jwoltjer/MD_simulations/base_files/multi_lipid/* ./ 
  
  #Copy forcefield files
  cp -r /gpfs/home1/jwoltjer/MD_simulations/run_dir/Slipids_2020.ff /gpfs/home1/jwoltjer/MD_simulations/run_dir/SLipids_2020 /gpfs/home1/jwoltjer/MD_simulations/run_dir/amber19sb.ff ./
  
#  # Run job here
#  echo "# [$(date)] job ${SLURM_JOB_NAME}"
#  echo "# [$(date)] Running Gromacs..."
  
  seed=76$a

  ###ACTUAL RUN COMMAND###
  
#  export GMX_MAXBACKUP=-1 #gets rit of #nvt.gro1# type files

  ### Build system ###
  
  gmx editconf -f system_gmx.gro  -o mol_z.gro -rotate 90 0 0 -box 7 7 7 -c
  
  cp system_gmx.top topol.top
  
  sed -i \
      -e '/\[ system \]/d' \
      -e '/; Name/d' \
      -e '/Generic title/d' \
      -e '/\[ molecules \]/d' \
      -e '/; Compound       #mols/d' \
      -e '/system1              1/d' \
      topol.top
  
  gmx insert-molecules -f mol_z.gro -ci system_gmx.gro -nmol 7 -o mol.gro  -try 10000 -seed $seed

  gmx insert-molecules -f mol.gro -ci POPC.pdb -nmol 64 -o mol_popc.gro  -try 100000 -seed $seed
  
  gmx insert-molecules -f mol_popc.gro -ci POPS.pdb -nmol 26 -o mol_popc_pops.gro  -try 100000 -seed $seed
  
  gmx insert-molecules -f mol_popc_pops.gro -ci cholesterol.pdb -nmol 38 -o mol_membrane.gro  -try 100000 -seed $seed

  HEADER='; Include forcefield parameters
#include "./amber19sb.ff/forcefield.itp"
#include "./Slipids_2020.ff/forcefield.itp"'
  FOOTER='
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
  
#include "./SLipids_2020/itp_files/POPC.itp"
#include "./SLipids_2020/itp_files/POPS.itp"
#include "./SLipids_2020/itp_files/cholesterol.itp"
  
[ system ]
; Name
Generic title in water
  
[ molecules ]
; Compound       #mols
system1              8
POPC                64
POPS                26
cholesterol         38'
  
  for file in topol.top; do
      sed -i '/1               2               yes             0.5          0.83333333  /d' "$file"
      # Create temp file with header, original content, and footer
      { echo "$HEADER"; cat "$file"; echo "$FOOTER"; } > "$file.tmp"
      mv "$file.tmp" "$file"
  done
  
  
  gmx solvate -cp mol_membrane.gro -cs spc216.gro -o mol_membrane_sol.gro -p topol.top
  
  gmx grompp -f ions.mdp -c  mol_membrane_sol.gro  -p topol.top -o ions.tpr -maxwarn 100
  
  echo 17 | gmx genion -s ions.tpr -o all.gro -p topol.top -pname NA -nname CL -rmin 0.1 -conc 0.15 -neutral
  
  echo -e "13 | 14 | 15\nname 23 Lipids\nname 1 Complex\n1 | 23\nq\n" | gmx make_ndx -f all.gro  

  

  gmx grompp -f em.mdp -c all.gro -p topol.top -o em.tpr -maxwarn 100
  gmx mdrun -v -deffnm em
  
  sleep 20 

  cp /gpfs/home1/jwoltjer/MD_simulations/base_files/*.sh ./
  sbatch --job-name="$(basename "$SLURM_SUBMIT_DIR")_$(basename "$PWD")" equi_multi_lipid.sh



  cd "${SLURM_SUBMIT_DIR}"

done
#!/bin/bash
#SBATCH --job-name=helix_calculation
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=m.j.woltjer@umail.leidenuniv.nl

set -u
shopt -s nullglob

module purge
module load 2025
module load GROMACS/2025.2-foss-2025a

base_dir="/gpfs/home1/jwoltjer/MD_simulations/results/equi/"
output_file="${base_dir}/H_remaining_percentages.tsv"

printf "system_folder\tnumbered_subfolder\tremaining_H_percent\tmean_percent\tstddev_percent\tmedian_percent\tmode_percent\tn_values\n" > "$output_file"

cd "$base_dir" || { echo "Cannot cd to $base_dir"; exit 1; }

run_dssp_with_fallback() {
    local structure_file="$1"
    local output_dat_file="$2"

    if gmx dssp -s "$structure_file" -n index.ndx -o "$output_dat_file" -sel Complex; then
        return 0
    fi

    echo "Complex selection failed for $structure_file, retrying with Protein"
    gmx dssp -s "$structure_file" -n index.ndx -o "$output_dat_file" -sel Protein
}

for system_path in "$base_dir"/double_bonded_rut_pep_protonated_cm5_multi_lipid_small/; do
    [ -d "$system_path" ] || continue
    system_name="$(basename "$system_path")"

    tmp_vals="$(mktemp)"
    count_vals=0

    for sub_path in "$system_path"*/; do
        [ -d "$sub_path" ] || continue
        sub_name="$(basename "$sub_path")"

        # Only process numbered folders: 1, 2, 20, ...
        [[ "$sub_name" =~ ^[0-9]+$ ]] || continue

        (
            cd "$sub_path" || exit 1

            # Check required files for first DSSP
            if [ ! -f "system_gmx.gro" ] || [ ! -f "index.ndx" ]; then
                printf "%s\t%s\tNA\tNA\tNA\tNA\tNA\tNA\n" "$system_name" "$sub_name" >> "$output_file"
                exit 0
            fi

            # First DSSP with fallback: Complex -> Protein
            if ! run_dssp_with_fallback "system_gmx.gro" "original_molecule.dat"; then
                printf "%s\t%s\tNA\tNA\tNA\tNA\tNA\tNA\n" "$system_name" "$sub_name" >> "$output_file"
                exit 0
            fi

            final_ok=0
            if [ -f "npt-equi2.gro" ]; then
                if run_dssp_with_fallback "npt-equi2.gro" "after_sim_molecule.dat"; then
                    final_ok=1
                fi
            fi

            # Build one combined helix file and remove only the two temporary helix files
            {
                echo "# HELIX DATA COMBINED"
                echo "# --- ORIGINAL (system_gmx.gro) ---"
                cat original_molecule.dat
                echo
                echo "# --- FINAL (npt-equi2.gro) ---"
                if [ "$final_ok" -eq 1 ]; then
                    cat after_sim_molecule.dat
                else
                    echo "# FINAL HELIX DATA NOT AVAILABLE"
                fi
            } > helix_combined.dat

            # Compute percentage only if final exists and original H count > 0
            if [ "$final_ok" -eq 1 ]; then
                h_original=$(tr -cd 'H' < original_molecule.dat | wc -c)
                h_final=$(tr -cd 'H' < after_sim_molecule.dat | wc -c)

                if [ "$h_original" -gt 0 ]; then
                    percent=$(awk -v f="$h_final" -v o="$h_original" 'BEGIN { printf "%.2f", (f/o)*100 }')
                    printf "%s\t%s\t%s\tNA\tNA\tNA\tNA\tNA\n" "$system_name" "$sub_name" "$percent" >> "$output_file"
                    echo "$percent" >> "$tmp_vals"
                else
                    printf "%s\t%s\tNA\tNA\tNA\tNA\tNA\tNA\n" "$system_name" "$sub_name" >> "$output_file"
                fi
            else
                printf "%s\t%s\tNA\tNA\tNA\tNA\tNA\tNA\n" "$system_name" "$sub_name" >> "$output_file"
            fi

            # Remove only the two temporary helix dat files, keep all other .dat files
            rm -f original_molecule.dat after_sim_molecule.dat
        )
    done

    if [ -s "$tmp_vals" ]; then
        count_vals=$(wc -l < "$tmp_vals")

        mean_std=$(awk '
            {x=$1; n++; sum+=x; sumsq+=x*x}
            END {
                if (n>0) {
                    mean=sum/n
                    var=(sumsq/n)-(mean*mean)
                    if (var<0) var=0
                    std=sqrt(var)
                    printf "%.2f\t%.2f", mean, std
                } else {
                    printf "NA\tNA"
                }
            }
        ' "$tmp_vals")

        median=$(sort -n "$tmp_vals" | awk -v n="$count_vals" '
            {a[NR]=$1}
            END {
                if (n==0) {print "NA"; exit}
                if (n%2==1) {
                    printf "%.2f", a[(n+1)/2]
                } else {
                    printf "%.2f", (a[n/2] + a[n/2 + 1]) / 2
                }
            }
        ')

        mode=$(awk '
            {
                k=sprintf("%.2f",$1)
                c[k]++
            }
            END {
                max=0
                mode="NA"
                for (k in c) {
                    if (c[k] > max) {
                        max=c[k]
                        mode=k
                    } else if (c[k] == max && mode != "NA" && (k+0) < (mode+0)) {
                        mode=k
                    }
                }
                print mode
            }
        ' "$tmp_vals")

        mean_val="${mean_std%%	*}"
        std_val="${mean_std##*	}"

        printf "%s\t__SUMMARY__\tNA\t%s\t%s\t%s\t%s\t%s\n" \
            "$system_name" "$mean_val" "$std_val" "$median" "$mode" "$count_vals" >> "$output_file"
    else
        printf "%s\t__SUMMARY__\tNA\tNA\tNA\tNA\tNA\t0\n" "$system_name" >> "$output_file"
    fi

    rm -f "$tmp_vals"
done

echo "Done. Results written to: $output_file"

import json
import os
import re
import subprocess
from datetime import datetime
import numpy as np

from bocoflow_core.logger import log_message
from bocoflow_core.node import Node, NodeException, NodeResult
from bocoflow_core.parameters import (
    BooleanParameter,
    FileParameterEdit,
    FolderParameter,
    StringParameter,
    IntegerParameter,
)


class MetalDockNode(Node):
    """
    Runs MetalDock using WSL with support for multiple metal complexes.

    MetalDock docks a specified ligand to a protein. The number of ligands to dock can be set.
    This node executes MetalDock by generating an INI configuration file which it uses along with given PDB and XYZ
    structure files. For multiple complexes, it iteratively docks each complex. After the first docking the docked to 
    LIG residue is temporarly converted to ALA to allow docking of subsequent complexes without interference.

    Input: PDB structure, XYZ coordinates, and configuration parameters
    Output: Clean protein PDB, ligand position PDB files, and merged PDB docked lignands
    """

    name = "MetalDock"
    node_key = "MetalDockNode"
    num_in = 0
    num_out = 1
    color = "#FF6B35"

    # Distance thresholds (Angstroms)
    H_HEAVY_BOND_MIN = 0.8
    H_HEAVY_BOND_MAX = 1.3
    HEAVY_HEAVY_BOND_MIN = 0.8
    HEAVY_HEAVY_BOND_MAX = 1.8
    H_LIG_CLASH_THRESHOLD = 2.0
    H_IDEAL_BOND_LENGTH = 1.0

    OPTIONS = {
        "case_name": StringParameter(
            "Case Name",
            docstring="Name of the case/protein for docking"
        ),
        "pdb_file": FileParameterEdit(
            "PDB Structure File",
            docstring="PDB file containing the protein structure"
        ),
        "xyz_file": FileParameterEdit(
            "XYZ Coordinates File",
            docstring="XYZ file containing coordinate data"
        ),
        "metal_symbol": StringParameter(
            "Metal Symbol",
            default="Re",
            docstring="Chemical symbol of the metal (e.g., Re, Fe, Cu)"
        ),
        "charge": IntegerParameter(
            "Charge",
            default=0,
            docstring="Charge of the metal complex"
        ),
        "spin": IntegerParameter(
            "Spin Multiplicity",
            default=0,
            docstring="Spin multiplicity of the metal complex"
        ),
        "vacant_site": BooleanParameter(
            "Vacant Site",
            default=True,
            docstring="Whether the metal complex has a vacant site"
        ),
        "num_metal_complexes": IntegerParameter(
            "Number of Metal Complexes",
            default=1,
            docstring="Number of metal complexes to dock sequentially (assumes same complex type)"
        ),
        "box_size": IntegerParameter(
            "Box Size",
            default=30,
            docstring="Size of the docking box in Angstroms"
        ),
        "output_dir": FolderParameter(
            "Output Directory",
            docstring="Output directory for docking results"
        ),
        "wsl_distro": StringParameter(
            "WSL Distribution Name",
            default="Ubuntu",
            docstring="Name of the WSL distribution to use"
        ),
        "metaldock_base_dir": StringParameter(
            "MetalDock Base Directory",
            docstring="Base directory path for MetalDock in WSL, e.g. /home/user/metaldock/MetalDock"
        ),
        "force_to_run": BooleanParameter(
            "Force to Run",
            default=False,
            docstring="If true, the node will be executed regardless of the database record"
        ),
    }

    def execute(self, predecessor_data, flow_vars):
        """Execute the MetalDock docking"""
        log_message(
            f"Starting execution of MetalDock for case: {flow_vars['case_name'].get_value()}"
        )
        try:
            # Initialize standard result
            result = NodeResult()
            result.metadata.update(
                {
                    "case_name": flow_vars["case_name"].get_value(),
                    "execution_time": datetime.now().isoformat(),
                }
            )

            # Get parameters
            case_name = flow_vars["case_name"].get_value()
            pdb_file = flow_vars["pdb_file"].get_value()
            xyz_file = flow_vars["xyz_file"].get_value()
            metal_symbol = flow_vars["metal_symbol"].get_value()
            charge = flow_vars["charge"].get_value()
            spin = flow_vars["spin"].get_value()
            vacant_site = flow_vars["vacant_site"].get_value()
            num_metal_complexes = flow_vars["num_metal_complexes"].get_value()
            output_dir = self.resolve_path(flow_vars["output_dir"].get_value())
            wsl_distro = flow_vars["wsl_distro"].get_value()
            metaldock_base = flow_vars["metaldock_base_dir"].get_value()
            box_size = flow_vars["box_size"].get_value()

            log_message(f"Resolved output directory: {output_dir}")
            log_message(f"Number of metal complexes to dock: {num_metal_complexes}")
            result.metadata.update({"output_dir": self.format_output_path(output_dir)})

            # Validate input files exist and get paths
            input_files = {
                "pdb": (self.resolve_path(pdb_file), "PDB"),
                "xyz": (self.resolve_path(xyz_file), "XYZ")
            }

            for file_path, file_type in input_files.values():
                if not os.path.exists(file_path):
                    raise NodeException("execution", f"{file_type} file not found: {file_path}")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Get filenames
            xyz_filename = os.path.basename(input_files["xyz"][0])

            # Track docking iterations
            all_iterations = []
            current_pdb_path = input_files["pdb"][0]
            cumulative_merged_pdb = None
            converted_lig_residues = []

            # Perform iterative docking
            for iteration in range(1, num_metal_complexes + 1):
                log_message(f"\n{'='*60}")
                log_message(f"Starting docking iteration {iteration}/{num_metal_complexes}")
                log_message(f"{'='*60}\n")

                try:
                    # Run docking iteration
                    iteration_result = self._run_docking_iteration(
                        iteration=iteration,
                        case_name=case_name,
                        pdb_path=current_pdb_path,
                        xyz_path=input_files["xyz"][0],
                        xyz_filename=xyz_filename,
                        metal_symbol=metal_symbol,
                        charge=charge,
                        spin=spin,
                        vacant_site=vacant_site,
                        output_dir=output_dir,
                        wsl_distro=wsl_distro,
                        metaldock_base=metaldock_base,
                        cumulative_merged_pdb=cumulative_merged_pdb,
                        box_size=box_size,
                    )

                    all_iterations.append(iteration_result)
                    
                    # Update cumulative merged PDB for next iteration
                    cumulative_merged_pdb = iteration_result["merged_pdb_path"]
                    converted_lig_residues.append(iteration_result["target_lig_resnum"])

                    # If this is not the last iteration, prepare PDB for next round
                    if iteration < num_metal_complexes:
                        log_message(f"Preparing PDB for iteration {iteration + 1}")
                        
                        # Convert the targeted LIG to ALA
                        next_pdb_path = os.path.join(
                            output_dir,
                            f"{case_name}_for_iteration_{iteration + 1}.pdb"
                        )
                        
                        self._convert_lig_to_ala(
                            input_pdb=cumulative_merged_pdb,
                            output_pdb=next_pdb_path,
                            target_lig_resnum=iteration_result["target_lig_resnum"]
                        )
                        
                        current_pdb_path = next_pdb_path
                        log_message(f"Created modified PDB for next iteration: {next_pdb_path}")

                except Exception as e:
                    log_message(f"ERROR in iteration {iteration}: {str(e)}")
                    log_message(f"Stopping at iteration {iteration-1} with {len(all_iterations)} successful docking(s)")
                    break

            # Check if we had any successful iterations
            if not all_iterations:
                raise NodeException(
                    "execution",
                    "No successful docking iterations completed"
                )

            log_message(f"\n{'='*60}")
            log_message(f"Completed {len(all_iterations)}/{num_metal_complexes} docking iterations")
            log_message(f"{'='*60}\n")

            # Record input files
            result.files["input"].update({
                "original_pdb": self.format_output_path(input_files["pdb"][0]),
                "xyz_file": self.format_output_path(input_files["xyz"][0])
            })

            # Compile all output files
            all_pdb_files = []
            for iter_result in all_iterations:
                all_pdb_files.extend(iter_result["pdb_files"])

            # Store docking details
            result.data = {
                "case_name": case_name,
                "input_files": {
                    "pdb": os.path.basename(input_files["pdb"][0]),
                    "xyz": xyz_filename,
                },
                "metal_configuration": {
                    "metal_symbol": metal_symbol,
                    "charge": charge,
                    "spin": spin,
                    "vacant_site": vacant_site,
                },
                "num_complexes_requested": num_metal_complexes,
                "num_complexes_docked": len(all_iterations),
                "iterations": [
                    {
                        "iteration": iter_result["iteration"],
                        "best_pose": iter_result["best_pose"],
                        "binding_energy": iter_result["best_energy"],
                        "target_lig_residue": iter_result["target_lig_resnum"],
                        "num_structures": len(iter_result["pdb_files"]),
                        "merged_pdb": os.path.relpath(iter_result["merged_pdb_path"], output_dir)
                    }
                    for iter_result in all_iterations
                ],
                "converted_lig_residues": converted_lig_residues,
                "working_path": output_dir,
            }

            # Set output files
            result.files["output"] = {
                f"structure_{i}": self.format_output_path(pdb['full_path'])
                for i, pdb in enumerate(all_pdb_files)
            }
            
            # Add all intermediate merged PDBs
            for iter_result in all_iterations:
                iter_num = iter_result["iteration"]
                result.files["output"][f"merged_w_{iter_num}_complexes"] = self.format_output_path(
                    iter_result["merged_pdb_path"]
                )

            # Final merged PDB
            result.files["output"]["final_merged"] = self.format_output_path(cumulative_merged_pdb)

            # Prepare result message
            result.success = True
            energy_summary = ", ".join([
                f"Complex {i+1}: {iter_result['best_energy']:.2f} kcal/mol"
                for i, iter_result in enumerate(all_iterations)
            ])
            result.message = (
                f"MetalDock completed successfully for {case_name}. "
                f"Docked {len(all_iterations)} metal complex(es). "
                f"Binding energies: {energy_summary}"
            )

            return result.to_json()

        except FileNotFoundError:
            log_message("WSL not found on system")
            raise NodeException(
                "metaldock execution",
                "WSL is not installed or not in PATH. Please install WSL to use this node."
            )
        except Exception as e:
            log_message(f"Error in MetalDockNode: {str(e)}")
            raise NodeException("metaldock execution", str(e))

    def _run_docking_iteration(self, iteration, case_name, pdb_path, xyz_path, xyz_filename,
                               metal_symbol, charge, spin, vacant_site, output_dir, box_size,
                               wsl_distro, metaldock_base, cumulative_merged_pdb):
        """
        Run a single docking iteration
        
        Returns:
            Dictionary with iteration results
        """
        pdb_filename = os.path.basename(pdb_path)
        
        # Generate INI file
        ini_content = self._generate_ini_content(
            pdb_filename=pdb_filename,
            xyz_filename=xyz_filename,
            metal_symbol=metal_symbol,
            charge=charge,
            spin=spin,
            vacant_site=vacant_site,
            box_size=box_size
        )

        # Create INI file
        ini_filename = f"{case_name}_iter_{iteration}.ini"
        ini_filepath = os.path.join(output_dir, ini_filename)
        
        log_message(f"Creating INI file: {ini_filepath}")
        with open(ini_filepath, 'w') as f:
            f.write(ini_content)

        # Setup WSL paths
        wsl_temp_dir = f"{metaldock_base}/BoCoFlow_iter_{iteration}"
        wsl_output_dir = self._convert_to_wsl_path(output_dir)
        
        log_message(f"WSL temporary directory: {wsl_temp_dir}")
        log_message(f"WSL output directory: {wsl_output_dir}")

        # Convert paths to WSL
        wsl_pdb_path = self._convert_to_wsl_path(pdb_path)
        wsl_xyz_path = self._convert_to_wsl_path(xyz_path)
        wsl_ini_file = self._convert_to_wsl_path(ini_filepath)

        # Create output subdirectory for this iteration
        iteration_output_dir = os.path.join(output_dir, f"iteration_{iteration}")
        os.makedirs(iteration_output_dir, exist_ok=True)

        # Combined command
        combined_command = (
            f"mkdir -p {wsl_temp_dir} && "
            f"cp {wsl_ini_file} {wsl_pdb_path} {wsl_xyz_path} {wsl_temp_dir}/ && "
            f"cd {wsl_temp_dir} && "
            f"pixi run -e default metaldock -i {ini_filename} -m dock && "
            f"mkdir -p {wsl_output_dir}/iteration_{iteration}/output && "
            f"cp -r {wsl_temp_dir}/output/* {wsl_output_dir}/iteration_{iteration}/output/ && "
            f"rm -rf {wsl_temp_dir}"
        )
        
        log_message(f"Executing MetalDock iteration {iteration}")

        metaldock_result = subprocess.run(
            ['wsl', '-d', wsl_distro, '--', 'bash', '-ic', combined_command],
            capture_output=True,
            text=True
        )

        # Log output
        if metaldock_result.stdout:
            log_message(f"MetalDock stdout (iteration {iteration}):\n{metaldock_result.stdout[:500]}")
        if metaldock_result.stderr:
            log_message(f"MetalDock stderr (iteration {iteration}):\n{metaldock_result.stderr[:500]}")

        if metaldock_result.returncode != 0:
            raise NodeException(
                "execution",
                f"MetalDock iteration {iteration} failed with return code {metaldock_result.returncode}"
            )

        log_message(f"MetalDock iteration {iteration} completed successfully")

        # Locate output files
        results_dir = os.path.join(iteration_output_dir, "output", "results")
        
        if not os.path.exists(results_dir):
            raise NodeException(
                "execution",
                f"Results directory not found for iteration {iteration}: {results_dir}"
            )

        # Find all PDB files
        pdb_files = []
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.pdb'):
                    full_path = os.path.join(root, file)
                    pdb_files.append({
                        'filename': file,
                        'relative_path': os.path.relpath(full_path, output_dir),
                        'full_path': full_path
                    })

        log_message(f"Found {len(pdb_files)} PDB files in iteration {iteration}")

        # Process best pose
        log_message(f"Processing best pose for iteration {iteration}")
        
        # Use cumulative merged PDB if available (for iteration 2+), otherwise use clean PDB
        protein_pdb_for_merging = cumulative_merged_pdb if cumulative_merged_pdb else None
        
        best_pose_number, best_energy, merged_pdb_path, target_lig_resnum = self._process_best_pose(
            results_dir=results_dir,
            output_dir=output_dir,
            metal_symbol=metal_symbol,
            case_name=case_name,
            iteration=iteration,
            cumulative_merged_pdb=protein_pdb_for_merging
        )

        log_message(f"Iteration {iteration} - Best pose: {best_pose_number}, Energy: {best_energy} kcal/mol, Target LIG: {target_lig_resnum}")

        return {
            "iteration": iteration,
            "best_pose": best_pose_number,
            "best_energy": best_energy,
            "target_lig_resnum": target_lig_resnum,
            "merged_pdb_path": merged_pdb_path,
            "pdb_files": pdb_files,
            "results_dir": results_dir
        }

    def _process_best_pose(self, results_dir, output_dir, metal_symbol, case_name, 
                        iteration, cumulative_merged_pdb=None):
        """
        Process the docking results to find the best pose and create a merged PDB file.
        
        Args:
            results_dir: Path to the results directory
            output_dir: Path to the output directory
            metal_symbol: Symbol of the metal used in docking
            case_name: Name of the case for output file naming
            iteration: Current iteration number
            cumulative_merged_pdb: Path to merged PDB from previous iterations (or None for first)
            
        Returns:
            Tuple of (pose_number, binding_energy, merged_pdb_path, target_lig_resnum)
        """
        # Find the DLG file
        dlg_file = self._find_file_in_dir(results_dir, '.dlg')
        log_message(f"Found DLG file: {dlg_file}")
        
        # Parse DLG file to find best pose
        best_pose, best_energy = self._parse_dlg_for_best_pose(dlg_file)
        log_message(f"Best pose from DLG: Run {best_pose} with energy {best_energy} kcal/mol")
        
        # Find the clean PDB file
        clean_pdb = self._find_file_in_dir(results_dir, '.pdb', prefix='clean_')
        log_message(f"Found clean PDB: {clean_pdb}")
        
        # Find the best pose PDB file
        pose_dir = os.path.join(results_dir, f"pose_{best_pose}")
        pose_pdb = self._find_file_in_dir(pose_dir, f'_{best_pose}.pdb')
        log_message(f"Found pose PDB: {pose_pdb}")
        
        # Read HETATM lines from pose PDB
        hetatm_lines = self._read_pdb_records(pose_pdb, 'HETATM')
        
        if not hetatm_lines:
            raise NodeException("post-processing", f"No HETATM lines found in pose PDB: {pose_pdb}")
        
        log_message(f"Found {len(hetatm_lines)} HETATM lines")
        
        # Extract metal coordinates
        metal_coords = self._extract_metal_coordinates(hetatm_lines, metal_symbol)
        log_message(f"Metal coordinates: {metal_coords}")
        
        # Determine which PDB to use for merging
        protein_pdb_to_use = cumulative_merged_pdb if cumulative_merged_pdb else clean_pdb
        log_message(f"Using {'cumulative merged' if cumulative_merged_pdb else 'clean'} PDB: {protein_pdb_to_use}")
        
        # Read protein PDB and find closest LIG residue
        protein_lines, lig_residues = self._parse_protein_pdb(protein_pdb_to_use)
        
        # Handle case where there are no LIG residues
        if not lig_residues:
            log_message("No LIG residues found in protein PDB - handling first metal docking")
            
            if iteration == 1:
                # For first iteration with no LIG, directly append metal complex as new LIG residue
                log_message("First metal docking: creating new LIG residue")
                
                # Find the highest residue number in protein
                max_resnum = self._find_max_residue_number(protein_lines)
                new_lig_resnum = max_resnum + 1
                
                log_message(f"Assigning metal complex to new LIG residue number: {new_lig_resnum}")
                
                # Update HETATM lines with new residue number and name as LIG
                updated_hetatm_lines = self._update_hetatm_residue_info(hetatm_lines, new_lig_resnum)
                
                # Append HETATM lines to protein (no insertion, just append)
                merged_lines = self._append_hetatm_to_protein(protein_lines, updated_hetatm_lines)
                
                # Renumber all atoms sequentially
                renumbered_lines = self._renumber_atoms(merged_lines)
                
                # Write merged PDB file
                merged_pdb_path = os.path.join(output_dir, f"{case_name}_merged_w_{iteration}_complexes.pdb")
                self._write_pdb_file(merged_pdb_path, renumbered_lines)
                
                log_message(f"Created merged PDB with first metal complex: {merged_pdb_path}")
                
                # No hydrogen correction needed for first metal (no adjacent LIG residues yet)
                log_message("Skipping hydrogen correction - no adjacent LIG residues")
                
                return best_pose, best_energy, merged_pdb_path, new_lig_resnum
            else:
                # For subsequent iterations with no LIG, cannot proceed
                log_message(f"ERROR: Cannot dock metal complex {iteration} - no LIG residues found for positioning")
                log_message("MetalDock requires at least one existing LIG residue to dock subsequent complexes")
                raise NodeException(
                    "post-processing",
                    f"Cannot dock metal complex {iteration}: No LIG residues found in PDB. "
                    "MetalDock can only dock one metal at a time. "
                    "To dock multiple metals, the PDB structure must contain LIG residues from previous docking iterations."
                )
        
        log_message(f"Found {len(lig_residues)} LIG residues")
        
        # Find closest LIG residue to metal
        closest_lig_resnum = self._find_closest_lig_residue(metal_coords, lig_residues)
        log_message(f"Closest LIG residue: {closest_lig_resnum}")
        
        # Update HETATM lines with correct residue number
        updated_hetatm_lines = self._update_hetatm_residue_info(hetatm_lines, closest_lig_resnum)
        
        # Merge protein and HETATM lines
        merged_lines = self._merge_protein_and_hetatm(protein_lines, updated_hetatm_lines, closest_lig_resnum)
        
        # Renumber all atoms sequentially
        renumbered_lines = self._renumber_atoms(merged_lines)
        
        # Write merged PDB file
        merged_pdb_path = os.path.join(output_dir, f"{case_name}_merged_w_{iteration}_complexes.pdb")
        self._write_pdb_file(merged_pdb_path, renumbered_lines)
        
        log_message(f"Created merged PDB: {merged_pdb_path}")
        
        # Correct hydrogen positions near LIG residues
        log_message(f"Correcting hydrogen positions near LIG residue {closest_lig_resnum}")
        self._correct_hydrogen_positions(merged_pdb_path, closest_lig_resnum)
        
        return best_pose, best_energy, merged_pdb_path, closest_lig_resnum

    def _find_max_residue_number(self, pdb_lines):
        """Find the maximum residue number in PDB lines"""
        max_resnum = 0
        
        for line in pdb_lines:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    resnum = int(line[22:26].strip())
                    max_resnum = max(max_resnum, resnum)
                except (ValueError, IndexError):
                    continue
        
        return max_resnum

    def _append_hetatm_to_protein(self, protein_lines, hetatm_lines):
        """Append HETATM lines to the end of protein lines (before TER/END)"""
        # Remove TER and END lines
        merged_lines = [line for line in protein_lines if not line.startswith(('TER', 'END'))]
        
        # Append HETATM lines
        merged_lines.extend(hetatm_lines)
        
        # Add TER and END
        merged_lines.extend(['TER', 'END'])
        
        return merged_lines

    def _correct_hydrogen_positions(self, pdb_path, target_lig_resnum):
        """
        Correct hydrogen positions in residues adjacent to the target LIG residue.
        Checks if hydrogens are too close (<2Å) to LIG atoms and repositions them appropriately.
        
        Args:
            pdb_path: Path to PDB file to correct (will be modified in place)
            target_lig_resnum: Residue number of the LIG residue
        """
        # Parse PDB file into structured atom data
        atoms = self._parse_pdb_atoms(pdb_path)
        
        # Get neighboring residue numbers
        neighbor_resnums = [target_lig_resnum - 1, target_lig_resnum + 1]
        
        # Get LIG heavy atoms (non-hydrogen)
        lig_heavy_atoms = [
            atom for atom in atoms 
            if atom['res_num'] == target_lig_resnum 
            and atom['res_name'] == 'LIG' 
            and atom['element'] != 'H'
        ]
        
        if not lig_heavy_atoms:
            return  # No LIG heavy atoms found
        
        # Find hydrogens that need correction
        hydrogens_to_correct = []
        
        for neighbor_resnum in neighbor_resnums:
            neighbor_hydrogens = [
                atom for atom in atoms 
                if atom['res_num'] == neighbor_resnum 
                and atom['element'] == 'H'
            ]
            
            for h_atom in neighbor_hydrogens:
                if self._is_too_close_to_lig(h_atom, lig_heavy_atoms, self.H_LIG_CLASH_THRESHOLD):
                    hydrogens_to_correct.append(h_atom)
        
        if hydrogens_to_correct:
            log_message(f"Found {len(hydrogens_to_correct)} hydrogens with LIG proximity issues")
            
            # Correct each hydrogen
            for h_atom in hydrogens_to_correct:
                # Find the center atom (heavy atom bonded to this hydrogen)
                center_atom = self._find_bonded_heavy_atom(h_atom, atoms)
                
                if not center_atom:
                    continue
                
                # Find other atoms bonded to the center atom
                bonded_atoms = self._find_bonded_atoms(center_atom, atoms, exclude_atom=h_atom)
                
                # Calculate new hydrogen position
                new_coords = self._calculate_hydrogen_position(center_atom, bonded_atoms, len(bonded_atoms))
                
                if new_coords:
                    old_coords = (h_atom['x'], h_atom['y'], h_atom['z'])
                    h_atom['x'], h_atom['y'], h_atom['z'] = new_coords
                    
                    # Determine geometry type for logging
                    geometry_type = {1: 'linear', 2: 'trigonal planar', 3: 'tetrahedral'}.get(
                        len(bonded_atoms), f'{len(bonded_atoms)}-bonded'
                    )
                    log_message(
                        f"  LIG-proximity: Corrected {h_atom['atom_name']} (res {h_atom['res_num']}) - "
                        f"Geometry: {geometry_type}, Bonded to: {center_atom['atom_name']}, "
                        f"Old: ({old_coords[0]:.3f}, {old_coords[1]:.3f}, {old_coords[2]:.3f}), "
                        f"New: ({new_coords[0]:.3f}, {new_coords[1]:.3f}, {new_coords[2]:.3f})"
                    )
        
        # After LIG-proximity corrections, validate ALL hydrogens in protein residues
        log_message("Validating all hydrogen bond lengths in protein residues")
        validation_corrections = self._validate_all_hydrogen_bonds(atoms)
        
        total_corrections = len(hydrogens_to_correct) + validation_corrections
        if total_corrections > 0:
            log_message(f"Total hydrogen corrections: {total_corrections} "
                       f"({len(hydrogens_to_correct)} LIG-proximity, {validation_corrections} general)")
            # Write corrected PDB
            self._write_pdb_from_atoms(pdb_path, atoms, self._read_pdb_lines(pdb_path))
        else:
            log_message("No hydrogen corrections needed")
    
    def _is_too_close_to_lig(self, h_atom, lig_heavy_atoms, threshold):
        """Check if hydrogen is too close to any LIG heavy atom"""
        hx, hy, hz = h_atom['x'], h_atom['y'], h_atom['z']
        
        for lig_atom in lig_heavy_atoms:
            lx, ly, lz = lig_atom['x'], lig_atom['y'], lig_atom['z']
            distance = np.linalg.norm([hx - lx, hy - ly, hz - lz])
            
            if distance < threshold:
                return True
        
        return False
    
    def _find_bonded_heavy_atom(self, h_atom, atoms):
        """
        Find the heavy atom bonded to a hydrogen atom.
        Uses distance and naming pattern matching.
        """
        hx, hy, hz = h_atom['x'], h_atom['y'], h_atom['z']
        h_res_num = h_atom['res_num']
        h_atom_name = h_atom['atom_name']
        
        # Get heavy atoms in the same residue (protein atoms only)
        same_res_heavy_atoms = [
            atom for atom in atoms 
            if atom['res_num'] == h_res_num 
            and atom['element'] != 'H'
            and atom['record'] == 'ATOM'
        ]
        
        # Find closest heavy atom within bonding distance
        candidates = []
        for heavy_atom in same_res_heavy_atoms:
            ax, ay, az = heavy_atom['x'], heavy_atom['y'], heavy_atom['z']
            dist = np.linalg.norm([hx - ax, hy - ay, hz - az])
            
            if self.H_HEAVY_BOND_MIN <= dist <= self.H_HEAVY_BOND_MAX:
                # Check naming pattern match (e.g., HB1 -> CB)
                score = self._calculate_name_match_score(h_atom_name, heavy_atom['atom_name'])
                candidates.append((dist, score, heavy_atom))
        
        if not candidates:
            return None
        
        # Sort by score (descending), then distance (ascending)
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return candidates[0][2]
    
    def _calculate_name_match_score(self, h_name, heavy_name):
        """Calculate score for atom name matching (higher is better)"""
        h_base = h_name.rstrip('0123456789')
        
        if len(h_base) >= 2 and h_base[0] == 'H':
            expected_heavy = h_base[1:]
            if heavy_name == expected_heavy or heavy_name.startswith(expected_heavy):
                return 2  # Perfect match
        
        return 0  # No match
    
    def _find_bonded_atoms(self, center_atom, atoms, exclude_atom=None):
        """
        Find atoms bonded to the center atom (excluding the specified atom).
        Uses distance-based criteria and allows cross-residue backbone bonds.
        """
        cx, cy, cz = center_atom['x'], center_atom['y'], center_atom['z']
        center_res_num = center_atom['res_num']
        center_atom_num = center_atom['atom_num']
        
        bonded = []
        
        for atom in atoms:
            # Skip excluded atom
            if atom is exclude_atom or atom['atom_num'] == center_atom_num:
                continue
            
            # Allow bonds to neighboring residues (±1) for backbone connectivity
            if abs(atom['res_num'] - center_res_num) > 1:
                continue
            
            # Skip hydrogens (we only want heavy atoms for geometry calculation)
            if atom['element'] == 'H':
                continue
            
            # Check distance
            ax, ay, az = atom['x'], atom['y'], atom['z']
            dist = np.linalg.norm([cx - ax, cy - ay, cz - az])
            
            if self.HEAVY_HEAVY_BOND_MIN <= dist <= self.HEAVY_HEAVY_BOND_MAX:
                bonded.append(atom)
        
        return bonded
    
    def _calculate_hydrogen_position(self, center_atom, bonded_atoms, num_bonded):
        """
        Calculate ideal hydrogen position based on molecular geometry.
        
        Geometry types:
        - 3 bonds: Tetrahedral (109.5°)
        - 2 bonds: Trigonal planar (120°)
        - 1 bond: Linear (180°)
        - 0 bonds: Cannot determine
        - >3 bonds: Average opposite direction
        """
        if num_bonded == 0:
            return None
        
        center = np.array([center_atom['x'], center_atom['y'], center_atom['z']])
        
        # Get normalized bond vectors
        bond_vectors = []
        for atom in bonded_atoms:
            vec = np.array([atom['x'], atom['y'], atom['z']]) - center
            bond_vectors.append(vec / np.linalg.norm(vec))
        
        # Calculate ideal direction for hydrogen based on geometry
        avg_vec = sum(bond_vectors) / len(bond_vectors)
        h_direction = -avg_vec
        
        # Normalize and scale to ideal bond length
        h_direction = h_direction / np.linalg.norm(h_direction)
        new_h_pos = center + h_direction * self.H_IDEAL_BOND_LENGTH
        
        return tuple(new_h_pos)
    
    def _validate_all_hydrogen_bonds(self, atoms):
        """
        Validate and correct all hydrogen bonds in protein residues (non-LIG).
        
        Checks:
        1. H bonded to only 1 heavy atom at distance 0.95-1.05 Å (ideal)
        2. H closer than 0.95 Å to 1 heavy atom → extend to 1.0 Å
        3. H bonded to 2+ heavy atoms within 1.3 Å → assign to best match, reposition
        
        Args:
            atoms: List of atom dictionaries
            
        Returns:
            Number of corrections made
        """
        corrections_made = 0
        
        # Get all protein hydrogens (exclude LIG and HETATM)
        protein_hydrogens = [
            atom for atom in atoms
            if atom['element'] == 'H' 
            and atom['res_name'] != 'LIG'
            and atom['record'] == 'ATOM'
        ]
        
        # Get all heavy atoms for distance checks
        heavy_atoms = [atom for atom in atoms if atom['element'] != 'H']
        
        for h_atom in protein_hydrogens:
            h_pos = np.array([h_atom['x'], h_atom['y'], h_atom['z']])
            
            # Find all heavy atoms within bonding distance (0.0 - 1.3 Å)
            nearby_heavy_atoms = []
            for heavy_atom in heavy_atoms:
                # Only check same residue for now
                if heavy_atom['res_num'] != h_atom['res_num']:
                    continue
                if heavy_atom['record'] != 'ATOM':  # Skip HETATM
                    continue
                
                heavy_pos = np.array([heavy_atom['x'], heavy_atom['y'], heavy_atom['z']])
                dist = np.linalg.norm(h_pos - heavy_pos)
                
                if 0.0 < dist <= 1.3:  # Exclude self (dist=0), include up to 1.3
                    nearby_heavy_atoms.append((dist, heavy_atom))
            
            if not nearby_heavy_atoms:
                continue  # No heavy atoms nearby, skip
            
            # Sort by distance
            nearby_heavy_atoms.sort(key=lambda x: x[0])
            
            # Case 1: Only 1 heavy atom within 1.3 Å
            if len(nearby_heavy_atoms) == 1:
                dist, heavy_atom = nearby_heavy_atoms[0]
                
                if dist < 0.95:
                    # Too short, extend to 1.0 Å
                    correction_type = "bond too short"
                    new_coords = self._extend_bond_to_length(h_atom, heavy_atom, 1.0)
                    if new_coords:
                        old_coords = (h_atom['x'], h_atom['y'], h_atom['z'])
                        h_atom['x'], h_atom['y'], h_atom['z'] = new_coords
                        corrections_made += 1
                        log_message(
                            f"  General: {h_atom['atom_name']} (res {h_atom['res_num']}) - {correction_type}, "
                            f"Bonded to: {heavy_atom['atom_name']}, "
                            f"Old dist: {dist:.3f} Å, New dist: 1.000 Å"
                        )
                # If 0.95 <= dist <= 1.05, it's fine, no correction needed
            
            # Case 2: Multiple heavy atoms within 1.3 Å (ambiguous bonding)
            elif len(nearby_heavy_atoms) >= 2:
                # Check if hydrogen is within ideal range (1.05 Å) of multiple atoms
                close_atoms = [(d, a) for d, a in nearby_heavy_atoms if d <= 1.05]
                
                if len(close_atoms) >= 2:
                    # Ambiguous: determine correct center atom
                    correction_type = "ambiguous bonding"
                    
                    # Use naming pattern and distance to ideal to pick best center atom
                    best_center = self._find_best_center_atom(h_atom, close_atoms)
                    
                    if best_center:
                        # Recalculate position based on geometry
                        bonded_atoms = self._find_bonded_atoms(best_center, atoms, exclude_atom=h_atom)
                        new_coords = self._calculate_hydrogen_position(best_center, bonded_atoms, len(bonded_atoms))
                        
                        if new_coords:
                            old_coords = (h_atom['x'], h_atom['y'], h_atom['z'])
                            h_atom['x'], h_atom['y'], h_atom['z'] = new_coords
                            corrections_made += 1
                            
                            geometry_type = {1: 'linear', 2: 'trigonal planar', 3: 'tetrahedral'}.get(
                                len(bonded_atoms), f'{len(bonded_atoms)}-bonded'
                            )
                            log_message(
                                f"  General: {h_atom['atom_name']} (res {h_atom['res_num']}) - {correction_type}, "
                                f"Geometry: {geometry_type}, Bonded to: {best_center['atom_name']}, "
                                f"Old: ({old_coords[0]:.3f}, {old_coords[1]:.3f}, {old_coords[2]:.3f}), "
                                f"New: ({new_coords[0]:.3f}, {new_coords[1]:.3f}, {new_coords[2]:.3f})"
                            )
        
        return corrections_made
    
    def _extend_bond_to_length(self, h_atom, heavy_atom, target_length):
        """
        Extend H-heavy bond to target length by moving hydrogen.
        
        Args:
            h_atom: Hydrogen atom dictionary
            heavy_atom: Heavy atom dictionary
            target_length: Target bond length in Angstroms
            
        Returns:
            New coordinates tuple or None
        """
        h_pos = np.array([h_atom['x'], h_atom['y'], h_atom['z']])
        heavy_pos = np.array([heavy_atom['x'], heavy_atom['y'], heavy_atom['z']])
        
        # Vector from heavy to hydrogen
        bond_vec = h_pos - heavy_pos
        current_length = np.linalg.norm(bond_vec)
        
        if current_length < 0.01:  # Avoid division by zero
            return None
        
        # Normalize and scale to target length
        new_h_pos = heavy_pos + (bond_vec / current_length) * target_length
        
        return tuple(new_h_pos)
    
    def _find_best_center_atom(self, h_atom, candidate_atoms):
        """
        Find the best center atom for a hydrogen from multiple candidates.
        Uses naming pattern matching and distance to ideal bond length.
        
        Args:
            h_atom: Hydrogen atom dictionary
            candidate_atoms: List of (distance, atom) tuples
            
        Returns:
            Best center atom dictionary or None
        """
        h_name = h_atom['atom_name']
        
        # Score each candidate
        scored_candidates = []
        for dist, heavy_atom in candidate_atoms:
            # Name match score (higher is better)
            name_score = self._calculate_name_match_score(h_name, heavy_atom['atom_name'])
            
            # Distance score: closer to 1.0 Å is better
            dist_score = 1.0 / (abs(dist - 1.0) + 0.01)  # Avoid division by zero
            
            # Combined score (weighted: name match is more important)
            total_score = name_score * 10 + dist_score
            
            scored_candidates.append((total_score, heavy_atom))
        
        if not scored_candidates:
            return None
        
        # Return atom with highest score
        scored_candidates.sort(key=lambda x: -x[0])
        return scored_candidates[0][1]

    def _convert_lig_to_ala(self, input_pdb, output_pdb, target_lig_resnum):
        """
        Convert a LIG residue to ALA by keeping only backbone (N, CA, C, O) and CB atoms,
        removing all hydrogens, and changing residue name to ALA.
        Also removes all non-LIG hydrogens from the entire structure.
        """
        log_message(f"Converting LIG residue {target_lig_resnum} to ALA")
        
        # Standard ALA atoms to keep
        ala_atoms = {'N', 'CA', 'C', 'O', 'CB'}
        
        output_lines = []
        hydrogens_removed = 0
        hydrogens_kept = 0
        
        with open(input_pdb, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    res_name = line[17:20].strip()
                    res_num = int(line[22:26].strip())
                    atom_name = line[12:16].strip()
                    element = line[76:78].strip() if len(line) > 77 else atom_name[0]
                    
                    # Check if this is the target LIG residue
                    if res_name == 'LIG' and res_num == target_lig_resnum:
                        # Only keep backbone and CB atoms (no hydrogens)
                        if atom_name in ala_atoms:
                            # Change residue name to ALA and record type to ATOM
                            new_line = (
                                'ATOM  ' +
                                line[6:17] +
                                'ALA' +
                                line[20:]
                            )
                            output_lines.append(new_line)
                    else:
                        # For all other atoms: remove hydrogens except LIG hydrogens
                        if element == 'H':
                            if res_name == 'LIG':
                                output_lines.append(line)
                                hydrogens_kept += 1
                            else:
                                hydrogens_removed += 1
                        else:
                            # Keep all non-hydrogen atoms
                            output_lines.append(line)
                else:
                    # Keep non-ATOM/HETATM lines
                    output_lines.append(line)
        
        # Renumber all atoms
        renumbered_lines = self._renumber_atoms(output_lines)
        
        # Write output
        self._write_pdb_file(output_pdb, renumbered_lines)
        log_message(f"Converted LIG {target_lig_resnum} to ALA in {output_pdb}")
        log_message(f"Removed {hydrogens_removed} non-LIG hydrogens, kept {hydrogens_kept} LIG hydrogens")

    def _parse_dlg_for_best_pose(self, dlg_file):
        """Parse DLG file to find the run with the lowest binding energy"""
        run_energies = {}
        current_run = None
        
        with open(dlg_file, 'r') as f:
            for line in f:
                # Look for run number
                run_match = re.search(r'Run:\s+(\d+)', line)
                if run_match:
                    current_run = int(run_match.group(1))
                
                # Look for binding energy for current run
                if current_run and 'Estimated Free Energy of Binding' in line:
                    energy_match = re.search(r'=\s+([-\d.]+)\s+kcal/mol', line)
                    if energy_match:
                        energy = float(energy_match.group(1))
                        run_energies[current_run] = energy
        
        if not run_energies:
            raise NodeException("post-processing", "Could not parse binding energies from DLG file")
        
        # Find run with lowest (most negative) energy
        best_run = min(run_energies, key=run_energies.get)
        best_energy = run_energies[best_run]
        
        return best_run, best_energy

    def _extract_metal_coordinates(self, hetatm_lines, metal_symbol):
        """Extract coordinates of the metal atom from HETATM lines"""
        for line in hetatm_lines:
            element = line[76:78].strip()
            if element == metal_symbol:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                return (x, y, z)
        
        raise NodeException("post-processing", f"Metal atom '{metal_symbol}' not found in HETATM lines")

    def _parse_protein_pdb(self, pdb_file):
        """Parse protein PDB file and extract LIG residues with their C atom positions"""
        protein_lines = []
        lig_residues = {}
        
        with open(pdb_file, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    protein_lines.append(line)
                    
                    # Check if this is a LIG residue
                    res_name = line[17:20].strip()
                    if res_name == 'LIG':
                        res_num = int(line[22:26].strip())
                        atom_name = line[12:16].strip()
                        
                        # Look for C atoms (CA, CB, CG, etc.)
                        if atom_name.startswith('C') and len(atom_name) >= 2:
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            
                            if res_num not in lig_residues:
                                lig_residues[res_num] = []
                            lig_residues[res_num].append((atom_name, x, y, z))
                elif line.startswith('TER') or line.startswith('END'):
                    protein_lines.append(line)
        
        # Validate that we found C atoms in LIG residues
        for resnum, atoms in lig_residues.items():
            if not atoms:
                raise NodeException(
                    "post-processing",
                    f"No suitable C atoms found in LIG residue {resnum}"
                )
        
        return protein_lines, lig_residues

    def _find_closest_lig_residue(self, metal_coords, lig_residues):
        """Find the LIG residue with a C atom closest to the metal"""
        min_distance = float('inf')
        closest_resnum = None
        
        mx, my, mz = metal_coords
        
        for resnum, atoms in lig_residues.items():
            for atom_name, x, y, z in atoms:
                distance = np.linalg.norm([x - mx, y - my, z - mz])
                if distance < min_distance:
                    min_distance = distance
                    closest_resnum = resnum
        
        if closest_resnum is None:
            raise NodeException("post-processing", "Could not determine closest LIG residue")
        
        log_message(f"Closest distance to metal: {min_distance:.3f} Å")
        
        return closest_resnum

    def _update_hetatm_residue_info(self, hetatm_lines, target_resnum):
        """Update HETATM lines to change UNK to LIG and set correct residue number"""
        return [
            line[:17] + 'LIG' + line[20:22] + f'{target_resnum:4d}' + line[26:]
            for line in hetatm_lines
        ]

    def _merge_protein_and_hetatm(self, protein_lines, hetatm_lines, target_resnum):
        """Merge protein and HETATM lines, inserting HETATMs after the target LIG residue"""
        # Remove TER and END lines
        merged_lines = [line for line in protein_lines if not line.startswith(('TER', 'END'))]
        
        # Find insertion point (after last atom of target LIG residue)
        inserted = False
        final_lines = []
        
        for i, line in enumerate(merged_lines):
            final_lines.append(line)
            
            if not inserted and (line.startswith('ATOM') or line.startswith('HETATM')):
                res_num = int(line[22:26].strip())
                res_name = line[17:20].strip()
                
                # Check if this is the target LIG residue
                if res_name == 'LIG' and res_num == target_resnum:
                    # Check if next line is different residue or end
                    is_last_in_residue = (
                        i + 1 >= len(merged_lines) or
                        int(merged_lines[i + 1][22:26].strip()) != target_resnum
                    )
                    
                    if is_last_in_residue:
                        final_lines.extend(hetatm_lines)
                        inserted = True
        
        # If not inserted, add at the end
        if not inserted:
            final_lines.extend(hetatm_lines)
        
        # Add TER and END
        final_lines.extend(['TER', 'END'])
        
        return final_lines

    def _renumber_atoms(self, lines):
        """Renumber all atoms sequentially starting from 1"""
        renumbered_lines = []
        atom_number = 1
        
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                new_line = line[:6] + f'{atom_number:5d}' + line[11:]
                renumbered_lines.append(new_line)
                atom_number += 1
            else:
                renumbered_lines.append(line)
        
        return renumbered_lines

    def _generate_ini_content(self, pdb_filename, xyz_filename, metal_symbol, charge, spin, vacant_site, box_size):
        """Generate INI file content for MetalDock"""

        return f"""[DEFAULT]
metal_symbol = {metal_symbol}
method = dock
ncpu = 16
memory = 8000

[PROTEIN]
pdb_file = {pdb_filename}
pH = 7.4
clean_pdb = True

[QM]
engine = ORCA
orcasimpleinput = B3LYP D3BJ def2-TZVP
orcablocks = %basis newECP Re "def2-SD" end end

[METAL_COMPLEX]
geom_opt = False
xyz_file = {xyz_filename}
charge = {charge}
spin = {spin}
vacant_site = {vacant_site}

[DOCKING]
rmsd = True
box_size = {box_size}
random_pos = True
num_poses = 50
"""

    def _convert_to_wsl_path(self, windows_path):
        """Convert Windows path to WSL path format"""
        if windows_path.startswith('/'):
            return windows_path
        
        if ':' in windows_path:
            drive = windows_path[0].lower()
            path = windows_path[2:].replace('\\', '/')
            return f"/mnt/{drive}{path}"
        
        return windows_path

    # ========== Helper Methods ==========
    
    def _find_file_in_dir(self, directory, suffix, prefix=''):
        """Find a file in directory with given suffix and optional prefix"""
        for file in os.listdir(directory):
            if file.startswith(prefix) and file.endswith(suffix):
                return os.path.join(directory, file)
        
        raise NodeException(
            "post-processing",
            f"No file with suffix '{suffix}' and prefix '{prefix}' found in {directory}"
        )
    
    def _read_pdb_records(self, pdb_file, record_type):
        """Read all lines of a specific record type from PDB file"""
        lines = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(record_type):
                    lines.append(line.rstrip('\n'))
        return lines
    
    def _read_pdb_lines(self, pdb_file):
        """Read all lines from PDB file"""
        with open(pdb_file, 'r') as f:
            return [line.rstrip('\n') for line in f]
    
    def _write_pdb_file(self, pdb_file, lines):
        """Write lines to PDB file"""
        with open(pdb_file, 'w') as f:
            for line in lines:
                f.write(line + '\n')
    
    def _parse_pdb_atoms(self, pdb_file):
        """Parse PDB file into structured atom data"""
        atoms = []
        for line in self._read_pdb_lines(pdb_file):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atoms.append({
                    'line': line,
                    'record': line[0:6].strip(),
                    'atom_num': int(line[6:11].strip()),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'res_num': int(line[22:26].strip()),
                    'x': float(line[30:38].strip()),
                    'y': float(line[38:46].strip()),
                    'z': float(line[46:54].strip()),
                    'element': line[76:78].strip() if len(line) > 77 else line[12:16].strip()[0]
                })
        return atoms
    
    def _write_pdb_from_atoms(self, pdb_file, atoms, original_lines):
        """Reconstruct PDB file from atom data, preserving non-ATOM/HETATM lines"""
        output_lines = []
        atom_idx = 0
        
        for line in original_lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_data = atoms[atom_idx]
                new_line = (
                    f"{atom_data['record']:<6}"
                    f"{atom_data['atom_num']:>5} "
                    f"{atom_data['atom_name']:<4}"
                    f"{line[16:30]}"
                    f"{atom_data['x']:>8.3f}"
                    f"{atom_data['y']:>8.3f}"
                    f"{atom_data['z']:>8.3f}"
                    f"{line[54:]}"
                )
                output_lines.append(new_line)
                atom_idx += 1
            else:
                output_lines.append(line)
        
        self._write_pdb_file(pdb_file, output_lines)

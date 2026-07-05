import json
import os
import subprocess
from datetime import datetime

from bocoflow_core.logger import log_message
from bocoflow_core.node import Node, NodeException, NodeResult
from bocoflow_core.parameters import (
    BooleanParameter,
    FileParameterEdit,
    FolderParameter,
    StringParameter,
)


class CoupleLigandPeptideNode(Node):
    """
    Couples a ligand to a peptide using OpenBabel in a conda environment.

    This node executes a Python script that couples ligand and peptide structures
    by running it in a specified conda environment. The script requires protein PDB,
    ligand PDB, coupling positions, and outputs a combined structure.

    Input: Protein PDB, Ligand PDB, positions
    Output: Coupled structure in PDB format
    """

    name = "Couple Ligand-Peptide (OpenBabel)"
    node_key = "CoupleLigandPeptide"
    num_in = 0
    num_out = 1
    color = "#FF6B6B"

    OPTIONS = {
        "case_name": StringParameter(
            "Case Name",
            docstring="Name of the case for tracking purposes"
        ),
        "protein_pdb": FileParameterEdit(
            "Protein PDB File",
            docstring="PDB file containing the protein/peptide structure"
        ),
        "ligand_pdb": FileParameterEdit(
            "Ligand PDB File",
            docstring="PDB file containing the ligand structure"
        ),
        "positions": StringParameter(
            "Coupling Positions",
            docstring="Positions where the ligand should be coupled (format depends on your script)"
        ),
        "output_dir": FolderParameter(
            "Output Directory",
            docstring="Directory where the output PDB file will be saved"
        ),
        "conda_location": FolderParameter(
            "Conda Installation Path",
            default="C:\\Users\\marti\\miniconda3",
            docstring="Path to your conda installation directory"
        ),
        "conda_env": StringParameter(
            "Conda Environment Name",
            default="bocoflow",
            docstring="Name of the conda environment to activate"
        ),
        "script_folder": FolderParameter(
            "Script Directory",
            default="C:\\Users\\marti\\Documents\\Study\\metallopeptide_thesis\\BoCoFlow\\node_tests\\nodes\\new_nodes",
            docstring="Directory containing the coupling script"
        ),
        "script_name": StringParameter(
            "Script Name",
            default="couple_ligand_peptide_openbabel_script_final.py",
            docstring="Name of the Python script to execute"
        ),
        "timeout": StringParameter(
            "Timeout (seconds)",
            default="600",
            docstring="Maximum execution time in seconds (default: 10 minutes)"
        ),
        "force_to_run": BooleanParameter(
            "Force to Run",
            default=False,
            docstring="If true, the node will be executed regardless of the database record"
        ),
    }

    def execute(self, predecessor_data, flow_vars):
        """Execute the ligand-peptide coupling script in conda environment"""
        log_message(
            f"Starting execution of CoupleLigandPeptide for case: {flow_vars['case_name'].get_value()}"
        )
        try:
            # Initialize result
            result = NodeResult()
            result.metadata.update(
                {
                    "case_name": flow_vars["case_name"].get_value(),
                    "execution_time": datetime.now().isoformat(),
                }
            )

            # Get parameters
            case_name = flow_vars["case_name"].get_value()
            protein_pdb = self.resolve_path(flow_vars["protein_pdb"].get_value())
            ligand_pdb = self.resolve_path(flow_vars["ligand_pdb"].get_value())
            positions = flow_vars["positions"].get_value()
            output_dir = self.resolve_path(flow_vars["output_dir"].get_value())
            conda_location = flow_vars["conda_location"].get_value()
            conda_env = flow_vars["conda_env"].get_value()
            script_folder = self.resolve_path(flow_vars["script_folder"].get_value())
            script_name = flow_vars["script_name"].get_value()
            timeout = int(flow_vars["timeout"].get_value())

            # Validate inputs
            if not os.path.exists(protein_pdb):
                raise NodeException("validation", f"Protein PDB file not found: {protein_pdb}")
            if not os.path.exists(ligand_pdb):
                raise NodeException("validation", f"Ligand PDB file not found: {ligand_pdb}")
            if not os.path.exists(script_folder):
                raise NodeException("validation", f"Script folder not found: {script_folder}")
            
            script_path = os.path.join(script_folder, script_name)
            if not os.path.exists(script_path):
                raise NodeException("validation", f"Script not found: {script_path}")

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Define output file path
            output_pdb = os.path.join(output_dir, f"{case_name}_coupled.pdb")

            log_message(f"Protein PDB: {protein_pdb}")
            log_message(f"Ligand PDB: {ligand_pdb}")
            log_message(f"Positions: {positions}")
            log_message(f"Output PDB: {output_pdb}")
            log_message(f"Using conda environment: {conda_env}")

            # Build the command
            conda_bat = os.path.join(conda_location, "Scripts", "activate.bat")
            if not os.path.exists(conda_bat):
                raise NodeException("validation", f"Conda activation script not found: {conda_bat}")

            # Build the full command chain
            # We use && to chain commands and /C to close after completion
            python_command = f'python "{script_name}" "{protein_pdb}" "{ligand_pdb}" "{positions}" "{output_pdb}"'
            
            full_command = (
                f'cmd /C "'
                f'"{conda_bat}" && '
                f'conda activate {conda_env} && '
                f'cd /d "{script_folder}" && '
                f'{python_command}"'
            )

            log_message(f"Executing command: {full_command}")

            # Execute the command
            process_result = subprocess.run(
                full_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Log output
            if process_result.stdout:
                log_message(f"Script stdout:\n{process_result.stdout}")
            if process_result.stderr:
                log_message(f"Script stderr:\n{process_result.stderr}")

            # Check for errors
            if process_result.returncode != 0:
                raise NodeException(
                    "execution",
                    f"Script execution failed with return code {process_result.returncode}:\n{process_result.stderr}"
                )

            # Verify output file was created
            if not os.path.exists(output_pdb):
                raise NodeException(
                    "execution",
                    f"Expected output file not created: {output_pdb}"
                )

            log_message(f"Coupling completed successfully: {output_pdb}")

            # Record input files
            result.files["input"].update(
                {
                    "protein_pdb": self.format_output_path(protein_pdb),
                    "ligand_pdb": self.format_output_path(ligand_pdb),
                    "script": self.format_output_path(script_path),
                }
            )

            # Store execution details
            result.data = {
                "case_name": case_name,
                "protein_pdb": os.path.basename(protein_pdb),
                "ligand_pdb": os.path.basename(ligand_pdb),
                "positions": positions,
                "conda_env": conda_env,
                "output_files": {
                    "coupled_structure": os.path.basename(output_pdb),
                    "coupled_structure_full_path": output_pdb,
                },
                "working_path": output_dir,
                "stdout": process_result.stdout[:500] if process_result.stdout else "",
            }

            # Set output files
            result.files["output"] = {
                "coupled_structure": self.format_output_path(output_pdb),
            }

            # Prepare result
            result.success = True
            result.message = f"Ligand-peptide coupling completed successfully for {case_name}"

            return result.to_json()

        except subprocess.TimeoutExpired:
            log_message(f"Script execution timed out after {timeout} seconds")
            raise NodeException(
                "execution",
                f"Script execution timed out after {timeout} seconds"
            )
        except Exception as e:
            log_message(f"Error in CoupleLigandPeptide: {str(e)}")
            raise NodeException("couple ligand peptide", str(e))
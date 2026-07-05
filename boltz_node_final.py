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


class Boltz2Prediction(Node):
    """
    Runs Boltz2 structure prediction using WSL.

    Boltz2 is a biomolecular structure prediction tool that runs in a WSL environment.
    This node executes Boltz2 predictions using a YAML configuration file which the node generates itself based on input parameters
    and returns the predicted structure in CIF format.

    Input: YAML configuration file
    Output: Predicted structure in CIF format
    """

    name = "Boltz2 Structure Prediction"
    node_key = "Boltz2Prediction"
    num_in = 0
    num_out = 1

    OPTIONS = {
        "case_name": StringParameter(
            "Case Name",
            docstring="Name of the case/protein for structure prediction"
        ),
        "sequence": StringParameter(
            "Amino Acid Sequence",
            docstring="Amino acid sequence using single letter codes (e.g., GWWLALALALALALALALALWWA)"
        ),
        "chain_id": StringParameter(
            "Chain ID",
            default="A",
            docstring="Chain identifier for the sequence"
        ),
        "output_dir": FolderParameter(
            "Output Directory",
            docstring="Output directory for prediction files (Windows path)"
        ),
        "wsl_distro": StringParameter(
            "WSL Distribution Name",
            default="Ubuntu",
            docstring="Name of the WSL distribution to use"
        ),
        "boltz_executable": StringParameter(
            "Boltz Executable Path",
            docstring="Full path to the boltz executable in WSL (e.g., /home/user/boltz2_env/.pixi/envs/default/bin/boltz)"
        ),
        # "use_msa_server": BooleanParameter(       #(temporarily disabled option)
        #     "Use MSA Server",
        #     default=True,
        #     docstring="Use MSA server for multiple sequence alignment"
        # ),
        # "no_kernels": BooleanParameter(       #(temporarily disabled option)  
        #     "No Kernels",
        #     default=True,
        #     docstring="Run without kernel optimizations (may be slower but more compatible)"
        # ),
        "force_to_run": BooleanParameter(
            "Force to Run",
            default=False,
            docstring="If true, the node will be executed regardless of the database record"
        ),
    }

    def execute(self, predecessor_data, flow_vars):
        """Execute the Boltz2 structure prediction"""
        log_message(
            f"Starting execution of Boltz2Prediction for case: {flow_vars['case_name'].get_value()}"
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
            output_dir = self.resolve_path(flow_vars["output_dir"].get_value())
            wsl_distro = flow_vars["wsl_distro"].get_value()
            boltz_executable = flow_vars["boltz_executable"].get_value()
            use_msa_server = True # flow_vars["use_msa_server"].get_value() #(temporarily disabled option)
            no_kernels = True # flow_vars["no_kernels"].get_value() #(temporarily disabled option)

            log_message(f"Resolved output directory: {output_dir}")
            result.metadata.update({"output_dir": self.format_output_path(output_dir)})

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Get sequence parameters
            sequence = flow_vars["sequence"].get_value()
            chain_id = flow_vars["chain_id"].get_value()

            # Generate YAML file
            yaml_filename = f"{case_name}.yaml"
            yaml_path = os.path.join(output_dir, yaml_filename)
            yaml_base = case_name

            yaml_content = f"""version: 1
sequences:
  - protein:
      id: {chain_id}
      sequence: {sequence}
"""

            with open(yaml_path, 'w') as f:
                f.write(yaml_content)

            log_message(f"Generated YAML file: {yaml_path}")

            # Convert paths to WSL format
            wsl_yaml_path = self._convert_to_wsl_path(yaml_path)
            wsl_output_dir = self._convert_to_wsl_path(output_dir)
            
            log_message(f"WSL YAML path: {wsl_yaml_path}")
            log_message(f"WSL output directory: {wsl_output_dir}")

            # Build command with WSL paths
            command_parts = [boltz_executable, "predict", wsl_yaml_path]
            
            if use_msa_server:
                command_parts.append("--use_msa_server")
            if no_kernels:
                command_parts.append("--no_kernels")
            
            # Add output directory
            command_parts.extend(["--out_dir", wsl_output_dir])
            
            command = " ".join(command_parts)
            log_message(f"Executing command: {command}")

            # Execute in WSL (no need to change directory)
            log_message(f"Running Boltz2 in WSL distribution: {wsl_distro}")
            wsl_result = subprocess.run(
                ['wsl', '-d', wsl_distro, '--', 'bash', '-c', command],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            # Log output
            if wsl_result.stdout:
                log_message(f"Boltz2 stdout:\n{wsl_result.stdout}")
            if wsl_result.stderr:
                log_message(f"Boltz2 stderr:\n{wsl_result.stderr}")
            
            if wsl_result.returncode != 0:
                raise NodeException(
                    "execution",
                    f"Boltz2 execution failed with return code {wsl_result.returncode}:\n{wsl_result.stderr}"
                )

            # Locate output CIF file (relative to output_dir)
            cif_relative_path = f"boltz_results_{yaml_base}/predictions/{yaml_base}/{yaml_base}_model_0.cif"
            output_cif_path = os.path.join(output_dir, cif_relative_path)

            if not os.path.exists(output_cif_path):
                raise NodeException(
                    "execution",
                    f"Expected output CIF file not found: {output_cif_path}"
                )

            log_message(f"Structure prediction completed: {output_cif_path}")

            # Record input files
            result.files["input"].update(
                {
                    "yaml_file": self.format_output_path(yaml_path),
                    "sequence": sequence,
                }
            )

            # Store prediction details with relative paths
            result.data = {
                "case_name": case_name,
                "sequence": sequence,  # ADD this line
                "chain_id": chain_id,
                "yaml_file": os.path.basename(yaml_path),
                "yaml_file_full_path": yaml_path,
                "wsl_distro": wsl_distro,
                "command": command,
                "output_files": {
                    "structure": cif_relative_path,  # Relative path
                    "structure_full_path": output_cif_path,  # Full path for reference
                },
                "working_path": output_dir,
                "stdout": wsl_result.stdout[:500] if wsl_result.stdout else "",  # First 500 chars
            }

            # Set output files with full paths for file tracking
            result.files["output"] = {
                "structure": self.format_output_path(output_cif_path),
            }

            # Prepare result
            result.success = True
            result.message = f"Boltz2 prediction completed successfully for {case_name}"

            return result.to_json()

        except subprocess.TimeoutExpired:
            log_message("Boltz2 execution timed out after 1 hour")
            raise NodeException(
                "boltz2 prediction",
                "Execution timed out after 1 hour"
            )
        except FileNotFoundError:
            log_message("WSL not found on system")
            raise NodeException(
                "boltz2 prediction",
                "WSL is not installed or not in PATH. Please install WSL to use this node."
            )
        except Exception as e:
            log_message(f"Error in Boltz2Prediction: {str(e)}")
            raise NodeException("boltz2 prediction", str(e))

    def _convert_to_wsl_path(self, windows_path):
        """Convert Windows path to WSL path format"""
        # Convert C:\Users\... to /mnt/c/Users/...
        if ':' in windows_path:
            drive = windows_path[0].lower()
            path = windows_path[2:].replace('\\', '/')
            return f"/mnt/{drive}{path}"
        return windows_path
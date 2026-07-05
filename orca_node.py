import json
import os
import subprocess
import shutil
from datetime import datetime
import re

from bocoflow_core.logger import log_message
from bocoflow_core.node import Node, NodeException, NodeResult
from bocoflow_core.parameters import (
    BooleanParameter,
    FileParameterEdit,
    FolderParameter,
    StringParameter,
    IntegerParameter,
)


class ORCAGeomOptNode(Node):
    """
    Runs ORCA using WSL.

    This node lets you run ORCA calculations from within BoCoFlow. It takes an XYZ structure file, the 
    desired commands, and other input parameters and uses these to generate an ORCA input file, executes the calculation in WSL, and retrieves the generated data.
    The terminal output is shown in real-time during the calculation.

    Requirements: 
    WSL must be installed on the system. 
    ORCA must be installed WSL.

    Input: XYZ structure file
    Output: Optimized XYZ geometry and all ORCA output files
    """

    name = "ORCA Geometry Optimization"
    node_key = "ORCAGeomOptNode"
    num_in = 0
    num_out = 1
    color = "#4A90E2"

    OPTIONS = {
        "case_name": StringParameter(
            "Case Name",
            docstring="Name of the case for the calculation"
        ),
        "xyz_file": FileParameterEdit(
            "XYZ Structure File",
            docstring="XYZ file containing the geometry optimized structure"
        ),
        "functional": StringParameter(
            "DFT Functional",
            default="B3LYP",
            docstring="DFT functional to use (e.g., B3LYP, PBE, M06-2X)"
        ),
        "dispersion": StringParameter(
            "Dispersion Correction",
            default="D4",
            docstring="Dispersion correction (e.g., D4, D3BJ, or leave empty for none)"
        ),
        "job_type": StringParameter(
            "Job Type",
            default="NUMFREQ",
            docstring="Type of calculation (e.g., OPT, FREQ, OPT FREQ)"
        ),
        "scf_convergence": StringParameter(
            "SCF Convergence",
            default="TIGHTSCF",
            docstring="SCF convergence criteria (e.g., TIGHTSCF, VERYTIGHTSCF)"
        ),
        "nprocs": IntegerParameter(
            "Number of Processors",
            default=16,
            docstring="Number of processors to use for the calculation"
        ),
        "charge": IntegerParameter(
            "Charge",
            default=0,
            docstring="Total charge of the system"
        ),
        "spin_multiplicity": IntegerParameter(
            "Spin Multiplicity",
            default=1,
            docstring="Spin multiplicity (2S+1, where S is total spin)"
        ),
        "metal_symbol": StringParameter( # this is needed incase the system contains metals that are not in the light basis set
            "Metal Symbol",
            default="Re",
            docstring="Chemical symbol of the metal atom (e.g., Re, Fe, Cu)"
        ),
        "light_basis": StringParameter(
            "Basis Set (Light Atoms)",
            default="6-31G(d)",
            docstring="Basis set for H, C, N, O, Cl atoms"
        ),
        "metal_basis": StringParameter(
            "Basis Set (Metal)",
            default="LANL2DZ",
            docstring="Basis set for the metal atom"
        ),
        "metal_ecp": StringParameter(
            "ECP (Metal)",
            default="HayWadt",
            docstring="Effective core potential for the metal atom"
        ),
        "output_dir": FolderParameter(
            "Output Directory",
            docstring="Output directory for calculation results (Windows path)"
        ),
        "orca_base_dir": StringParameter(
            "ORCA Base Directory",
            docstring="Base directory for ORCA calculations in WSL (e.g., /home/jacob/ambertools)"
        ),
        "orca_command": StringParameter(
            "ORCA Command",
            default="/usr/local/orca_6_1_0/orca",
            docstring="Full ORCA command/path (e.g., /usr/local/orca_6_1_0/orca or)"
        ),
        "wsl_distro": StringParameter(
            "WSL Distribution Name",
            default="Ubuntu",
            docstring="Name of the WSL distribution to use"
        ),
        "force_to_run": BooleanParameter(
            "Force to Run",
            default=False,
            docstring="If true, the node will be executed regardless of the database record"
        ),
    }

    def execute(self, predecessor_data, flow_vars):
        """Execute the ORCA geometry optimization"""
        log_message(
            f"Starting execution of ORCA Geometry Optimization for case: {flow_vars['case_name'].get_value()}"
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
            xyz_file = flow_vars["xyz_file"].get_value()
            output_dir = flow_vars["output_dir"].get_value()
            orca_base_dir = flow_vars["orca_base_dir"].get_value()
            orca_command = flow_vars["orca_command"].get_value()
            wsl_distro = flow_vars["wsl_distro"].get_value()
            
            functional = flow_vars["functional"].get_value()
            dispersion = flow_vars["dispersion"].get_value()
            job_type = flow_vars["job_type"].get_value()
            scf_convergence = flow_vars["scf_convergence"].get_value()
            nprocs = flow_vars["nprocs"].get_value()
            charge = flow_vars["charge"].get_value()
            spin_multiplicity = flow_vars["spin_multiplicity"].get_value()
            metal_symbol = flow_vars["metal_symbol"].get_value()
            light_basis = flow_vars["light_basis"].get_value()
            metal_basis = flow_vars["metal_basis"].get_value()
            metal_ecp = flow_vars["metal_ecp"].get_value()

            log_message(f"Input XYZ file: {xyz_file}")
            log_message(f"Output directory: {output_dir}")
            log_message(f"ORCA base directory: {orca_base_dir}")
            log_message(f"ORCA command: {orca_command}")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            result.metadata.update({"output_dir": output_dir})

            # Parse XYZ file
            log_message("Parsing XYZ file")
            coordinates = self._parse_xyz_file(xyz_file)
            log_message(f"Found {len(coordinates)} atoms")

            # Generate ORCA input file content
            log_message("Generating ORCA input file")
            inp_content = self._generate_orca_input(
                functional=functional,
                dispersion=dispersion,
                job_type=job_type,
                scf_convergence=scf_convergence,
                nprocs=nprocs,
                charge=charge,
                spin_multiplicity=spin_multiplicity,
                coordinates=coordinates,
                metal_symbol=metal_symbol,
                light_basis=light_basis,
                metal_basis=metal_basis,
                metal_ecp=metal_ecp
            )

            # Create input file
            inp_filename = f"geomopt_{case_name}.inp"
            inp_filepath = os.path.join(output_dir, inp_filename)
            
            log_message(f"Creating input file: {inp_filepath}")
            with open(inp_filepath, 'w') as f:
                f.write(inp_content)

            # Setup WSL paths
            # Ensure orca_base_dir doesn't have trailing slash
            orca_base_dir = orca_base_dir.rstrip('/')
            
            wsl_temp_dir = f"{orca_base_dir}/BoCoFlow_orca_{case_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wsl_output_dir = self._convert_to_wsl_path(output_dir)
            wsl_inp_file = self._convert_to_wsl_path(inp_filepath)

            log_message(f"WSL temporary directory: {wsl_temp_dir}")
            log_message(f"WSL output directory: {wsl_output_dir}")

            # Output filename
            output_filename = f"geomopt_{case_name}.out"
            
            # Combined command: create temp dir, copy input, run ORCA, copy outputs, cleanup, keep shell open
            combined_command = (
                f"mkdir -p {wsl_temp_dir} && "
                f"cp {wsl_inp_file} {wsl_temp_dir}/ && "
                f"cd {wsl_temp_dir} && "
                f"{orca_command} {inp_filename} 2>&1 | tee {output_filename} && "
                f"cp -r {wsl_temp_dir}/* {wsl_output_dir}/ && "
                f"rm -rf {wsl_temp_dir}"
            )

            log_message(f"Executing ORCA calculation")
            log_message(f"Command: {combined_command}")
            log_message("=" * 80)
            log_message("ORCA calculation started - check the terminal window")
            log_message("=" * 80)

            # Use Popen WITH CREATE_NEW_CONSOLE to show terminal window
            process = subprocess.Popen(
                ['wsl', '-d', wsl_distro, '--', 'bash', '-ic', combined_command],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )

            # Wait for process to complete
            return_code = process.wait()

            log_message("=" * 80)
            log_message(f"ORCA calculation finished with return code: {return_code}")
            log_message("=" * 80)

            if return_code != 0:
                raise NodeException(
                    "execution",
                    f"ORCA calculation failed with return code {return_code}"
                )

            log_message("ORCA calculation completed successfully")

            # Find optimized geometry file
            optimized_xyz = self._find_optimized_geometry(output_dir, case_name)
            
            if optimized_xyz:
                log_message(f"Found optimized geometry: {optimized_xyz}")
            else:
                log_message("Warning: Optimized geometry file not found")

            # Prepare result
            result.data = {
                "case_name": case_name,
                "input_xyz": xyz_file,
                "input_file": inp_filepath,
                "output_file": os.path.join(output_dir, output_filename),
                "optimized_xyz": optimized_xyz,
                "working_path": output_dir,
            }

            # Set output files
            result.files["input"] = {
                "xyz_file": xyz_file,
                "inp_file": inp_filepath,
            }
            
            result.files["output"] = {
                "output_file": os.path.join(output_dir, output_filename),
            }
            
            if optimized_xyz:
                result.files["output"]["optimized_xyz"] = optimized_xyz

            result.success = True
            result.message = f"ORCA geometry optimization completed successfully for {case_name}"

            return result.to_json()

        except FileNotFoundError:
            log_message("WSL not found on system")
            raise NodeException(
                "orca calculation",
                "WSL is not installed or not in PATH. Please install WSL to use this node."
            )
        except Exception as e:
            log_message(f"Error in ORCA Geometry Optimization: {str(e)}")
            raise NodeException("orca calculation", str(e))

    def _parse_xyz_file(self, xyz_file):
        """
        Parse XYZ file and extract element symbols and coordinates.
        Skips the first two lines (atom count and empty line) and extracts
        only element symbol and x, y, z coordinates.
        
        Returns:
            List of tuples: [(element, x, y, z), ...]
        """
        coordinates = []
        
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
        
        # Skip first line (atom count) and second line (empty)
        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                element = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                coordinates.append((element, x, y, z))
        
        return coordinates

    def _generate_orca_input(self, functional, dispersion, job_type, scf_convergence,
                            nprocs, charge, spin_multiplicity, coordinates,
                            metal_symbol, light_basis, metal_basis, metal_ecp):
        """
        Generate ORCA input file content based on parameters and coordinates.
        """
        # Build the main command line
        command_parts = [f"! {functional}"]
        
        if dispersion:
            command_parts.append(dispersion)
        
        command_parts.append(job_type)
        command_parts.append(scf_convergence)
        
        command_line = " ".join(command_parts)
        
        # Start building the input file
        inp_content = f"{command_line}\n\n"
        
        # Add processor specification
        inp_content += f"%pal\n  nprocs {nprocs}\nend\n\n"
        
        # Add basis set specifications
        inp_content += "%basis\n"
        
        # Light atoms basis sets
        for element in ['H', 'C', 'O', 'N', 'Cl']:
            inp_content += f'  NewGTO {element} \n'
            inp_content += f'  "{light_basis}"\n'
            inp_content += '  end\n'
        
        # Metal basis set
        inp_content += f'  NewGTO {metal_symbol} \n'
        inp_content += f'  "{metal_basis}"\n'
        inp_content += '  end\n'
        
        # Metal ECP
        inp_content += f'  NewECP {metal_symbol}\n'
        inp_content += f'  "{metal_ecp}"\n'
        inp_content += '  end\n'
        
        inp_content += 'end\n\n'
        
        # Add coordinates
        inp_content += f"* xyz {charge} {spin_multiplicity}\n"
        
        for element, x, y, z in coordinates:
            inp_content += f"{element:<4} {x:12.6f} {y:12.6f} {z:12.6f}\n"
        
        inp_content += "*\n"
        
        return inp_content

    def _find_optimized_geometry(self, output_dir, case_name):
        """
        Find the optimized geometry XYZ file in the output directory.
        ORCA typically creates files like: casename.xyz or casename_trj.xyz
        """
        possible_names = [
            f"geomopt_{case_name}.xyz",
            f"geomopt_{case_name}_trj.xyz",
        ]
        
        for filename in possible_names:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                return filepath
        
        # If not found with expected names, search for any .xyz file
        for file in os.listdir(output_dir):
            if file.endswith('.xyz') and 'geomopt' in file:
                return os.path.join(output_dir, file)
        
        return None

    def _convert_to_wsl_path(self, windows_path):
        """Convert Windows path to WSL path format"""
        # Handle UNC paths (\\wsl.localhost\...)
        if windows_path.startswith('\\\\wsl.localhost\\'):
            # Extract distribution and path
            parts = windows_path.replace('\\\\wsl.localhost\\', '').split('\\', 1)
            if len(parts) == 2:
                # Skip distribution name, use absolute path
                return '/' + parts[1].replace('\\', '/')
            return windows_path.replace('\\', '/')
        
        # Handle regular Windows paths (C:\...)
        if ':' in windows_path:
            drive = windows_path[0].lower()
            path = windows_path[2:].replace('\\', '/')
            return f"/mnt/{drive}{path}"
        
        return windows_path
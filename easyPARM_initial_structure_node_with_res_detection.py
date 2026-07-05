import json
import os
import subprocess
import shutil
from datetime import datetime
import numpy as np

from bocoflow_core.logger import log_message
from bocoflow_core.node import Node, NodeException, NodeResult
from bocoflow_core.parameters import (
    BooleanParameter,
    FileParameterEdit,
    FolderParameter,
    StringParameter,
)


class EasyPARMPreparation(Node):
    """
    Prepares protein structures using easyPARM in WSL.

    easyPARM is a tool for generating force field parameters for metalorganic complexes.
    This node handles the generation of the initial structure needed for DFT calculations.
    The node also handles complexes with multiple metal centres.
    
    The node can process structures with or without LIG residues:
    - With LIG residues: Creates separate XYZ files for each LIG residue
    - Without LIG residues: Processes the entire structure through easyPARM
    
    The node automatically detects and adds hydrogens using reduce if they are missing.

    Requirements: 
    WSL must be installed on the system. 
    Install easyPARM in a WSL environment. How to can be found at: https://abdelazim-abdelgawwad.github.io/Tutorial/Installation
    AmberTools (with reduce) should be available in the WSL environment.
    
    Input: PDB structure file
    Output: initial_structure.xyz files for each LIG residue (or for the whole structure if no LIG)
    """

    name = "easyPARM Structure Preparation"
    node_key = "EasyPARMPreparation"
    num_in = 0
    num_out = 1
    color = "#9B59B6"

    OPTIONS = {
        "case_name": StringParameter(
            "Case Name",
            docstring="Name of the case/protein for structure preparation"
        ),
        "pdb_file": FileParameterEdit(
            "PDB Structure File",
            docstring="PDB file containing the protein structure with or without LIG residue(s)"
        ),
        "output_dir": FolderParameter(
            "Output Directory",
            docstring="Output directory for prepared structure files (Windows path)"
        ),
        "easyparm_dir": StringParameter(
            "easyPARM Directory",
            docstring="Full WSL path to the easyPARM directory (e.g., /home/jacob/ambertools/)"
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
        """Execute the easyPARM structure preparation"""
        log_message(
            f"Starting execution of EasyPARMPreparation for case: {flow_vars['case_name'].get_value()}"
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
            output_dir = flow_vars["output_dir"].get_value()
            easyparm_dir = flow_vars["easyparm_dir"].get_value()
            wsl_distro = flow_vars["wsl_distro"].get_value()
            
             # Check if input is XYZ file and convert to PDB
            if pdb_file.lower().endswith('.xyz'):
                log_message(f"Input file is XYZ format. Converting to PDB...")
                pdb_file = self._convert_xyz_to_pdb(pdb_file, output_dir, case_name)
                log_message(f"Converted to PDB: {pdb_file}")

            #Check and identify amino acids in the structure
            pdb_file = self._check_and_identify_residues(pdb_file, output_dir, case_name)

            log_message(f"Input PDB file: {pdb_file}")
            log_message(f"Output directory: {output_dir}")
            log_message(f"easyPARM directory: {easyparm_dir}")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            result.metadata.update({"output_dir": output_dir})

            # Check if structure has hydrogens
            has_hydrogens = self._check_for_hydrogens(pdb_file)
            
            if not has_hydrogens:
                log_message("No hydrogens detected in structure. Adding hydrogens with reduce...")
                pdb_file = self._add_hydrogens(pdb_file, output_dir, case_name, wsl_distro, easyparm_dir)
                log_message(f"Hydrogens added. New PDB file: {pdb_file}")
            else:
                log_message("Hydrogens detected in structure. Proceeding without protonation.")

            # Parse PDB to find all LIG residues
            lig_residues = self._find_lig_residues(pdb_file)
            
            if not lig_residues:
                log_message("No LIG residues found. Processing entire structure through easyPARM.")
                
                # Process entire structure
                xyz_file = self._run_easyparm_no_lig(
                    pdb_file,
                    case_name,
                    output_dir,
                    easyparm_dir,
                    wsl_distro
                )
                
                # Prepare result for no LIG case
                result.data = {
                    "case_name": case_name,
                    "input_pdb": pdb_file,
                    "lig_residues": [],
                    "output_files": [{
                        "lig_resnum": None,
                        "xyz_file": xyz_file
                    }],
                    "working_path": output_dir,
                    "had_hydrogens": has_hydrogens,
                }
                
                result.files["input"] = {
                    "pdb_file": pdb_file,
                }
                
                result.files["output"] = {
                    "structure_xyz": xyz_file
                }
                
                result.success = True
                result.message = f"easyPARM preparation completed successfully for {case_name} (no LIG residues)"
                
                return result.to_json()

            log_message(f"Found {len(lig_residues)} LIG residue(s): {lig_residues}")

            # Process each LIG residue
            output_xyz_files = []
            
            for lig_resnum in lig_residues:
                log_message(f"Processing LIG residue {lig_resnum}")
                
                # Generate PDB with only this LIG residue
                if len(lig_residues) > 1:
                    temp_pdb = self._create_single_lig_pdb(
                        pdb_file, lig_resnum, lig_residues, output_dir, case_name
                    )
                else:
                    # If only one LIG, use original PDB
                    temp_pdb = pdb_file
                
                # Run easyPARM for this PDB
                xyz_file = self._run_easyparm(
                    temp_pdb, 
                    lig_resnum, 
                    case_name, 
                    output_dir, 
                    easyparm_dir, 
                    wsl_distro
                )
                
                output_xyz_files.append({
                    "lig_resnum": lig_resnum,
                    "xyz_file": xyz_file
                })
                
                # Clean up temporary PDB if created
                if len(lig_residues) > 1 and os.path.exists(temp_pdb):
                    os.remove(temp_pdb)
                    log_message(f"Removed temporary PDB: {temp_pdb}")

            # Prepare result
            result.data = {
                "case_name": case_name,
                "input_pdb": pdb_file,
                "lig_residues": lig_residues,
                "output_files": output_xyz_files,
                "working_path": output_dir,
                "had_hydrogens": has_hydrogens,
            }

            # Set output files
            result.files["input"] = {
                "pdb_file": pdb_file,
            }
            
            result.files["output"] = {
                f"LIG_{item['lig_resnum']}_xyz": item['xyz_file']
                for item in output_xyz_files
            }

            result.success = True
            result.message = f"easyPARM preparation completed successfully for {case_name} with {len(lig_residues)} LIG residue(s)"

            return result.to_json()

        except FileNotFoundError:
            log_message("WSL not found on system")
            raise NodeException(
                "easyparm preparation",
                "WSL is not installed or not in PATH. Please install WSL to use this node."
            )
        except Exception as e:
            log_message(f"Error in EasyPARMPreparation: {str(e)}")
            raise NodeException("easyparm preparation", str(e))

    def _convert_xyz_to_pdb(self, xyz_file, output_dir, case_name):
        """Convert XYZ file to PDB format"""
        import os
        
        log_message(f"Converting XYZ file to PDB: {xyz_file}")
        
        pdb_file = os.path.join(output_dir, f"{case_name}_from_xyz.pdb")
        
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
        
        # Skip first two lines (atom count and comment)
        atom_lines = lines[2:] if len(lines) > 2 else lines
        
        with open(pdb_file, 'w') as f:
            atom_num = 1
            for line in atom_lines:
                parts = line.split()
                if len(parts) >= 4:
                    element = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    
                    # Write PDB ATOM line
                    # Format: ATOM serial name resName chainID resSeq x y z occupancy tempFactor element
                    pdb_line = f"ATOM  {atom_num:5d}  {element:<3s} UNK A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
                    f.write(pdb_line)
                    atom_num += 1
            
            f.write("END\n")
        
        log_message(f"Successfully converted XYZ to PDB: {pdb_file}")
        return pdb_file
    
    def _check_and_identify_residues(self, pdb_file, output_dir, case_name):
        """
        Check if residues are already properly named. If not, identify amino acids
        by finding backbone patterns and analyzing atoms between them.
        """
        log_message("Checking and identifying amino acid residues...")
        
        # Standard amino acid names
        STANDARD_AA = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
        }
        
        log_message("Reading PDB file for residue identification...")
        
        # Read PDB and check current naming
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
        
        # Check if residues are already properly named
        residues_found = set()
        all_properly_named = True
        
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                res_name = line[17:20].strip()
                residues_found.add(res_name)
                if res_name not in STANDARD_AA and res_name != 'LIG':
                    all_properly_named = False
        
        if all_properly_named and residues_found:
            log_message("Residues are already properly named with standard amino acids. Skipping identification.")
            return pdb_file
        
        log_message("Residues are not properly named. Identifying amino acids by backbone pattern...")
        
        # Parse atoms and identify residues
        identified_pdb = os.path.join(output_dir, f"{case_name}_identified.pdb")
        self._identify_residues_by_backbone(pdb_file, identified_pdb)
        
        log_message(f"Amino acid identification complete. Output: {identified_pdb}")
        return identified_pdb
    
    def _identify_residues_by_backbone(self, input_pdb, output_pdb):
        """
        Identify residues by finding backbone patterns (N-C-C-O or N-CA-C-O)
        and analyzing the atoms between consecutive backbones.
        """
        import numpy as np
        
        # Read all atoms from PDB
        atoms = []
        with open(input_pdb, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    atoms.append({
                        'line': line,
                        'serial': int(line[6:11].strip()) if line[6:11].strip() else len(atoms) + 1,
                        'name': line[12:16].strip(),
                        'resName': line[17:20].strip(),
                        'chainID': line[21:22].strip() if len(line) > 21 else '',
                        'resSeq': int(line[22:26].strip()) if line[22:26].strip() else 1,
                        'x': float(line[30:38].strip()),
                        'y': float(line[38:46].strip()),
                        'z': float(line[46:54].strip()),
                        'element': line[76:78].strip() if len(line) > 77 else line[12:16].strip()[0],
                    })
        
        log_message(f"Read {len(atoms)} atoms from PDB")
        
        # Find backbone atoms (N, C, C, O patterns)
        backbone_groups = self._find_backbone_groups(atoms)
        
        if not backbone_groups:
            log_message("No backbone patterns found. Naming entire structure as LIG")
            for atom in atoms:
                atom['resName'] = 'LIG'
        else:
            log_message(f"Found {len(backbone_groups)} backbone groups")
            
            # Assign each atom to a residue based on backbone groups
            self._assign_atoms_to_residues(atoms, backbone_groups)
        
        # Write output PDB
        self._write_identified_pdb(atoms, output_pdb)
    
    def _find_backbone_groups(self, atoms):
        """
        Find groups of atoms forming backbone patterns (N-C-C-O).
        First tries simple sequential search, then falls back to distance-based search.
        Returns list of backbone groups with their atom indices and positions.
        """
        import numpy as np
        
        log_message("Searching for backbone N-C-C-O patterns...")
        
        # First attempt: Simple sequential search - look for N, C, C, O in order
        backbone_groups = []
        
        i = 0
        while i < len(atoms) - 3:
            # Check if we have N-C-C-O pattern in sequence
            if (atoms[i]['element'] == 'N' and
                atoms[i+1]['element'] == 'C' and
                atoms[i+2]['element'] == 'C' and
                atoms[i+3]['element'] == 'O'):
                
                # Found a backbone pattern
                avg_pos = (atoms[i]['serial'] + atoms[i+1]['serial'] + 
                          atoms[i+2]['serial'] + atoms[i+3]['serial']) / 4
                
                backbone_groups.append({
                    'N': i,
                    'CA': i+1,
                    'C': i+2,
                    'O': i+3,
                    'position': avg_pos
                })
                
                log_message(f"Found backbone at indices {i}-{i+3}: N={atoms[i]['serial']}, "
                          f"CA={atoms[i+1]['serial']}, C={atoms[i+2]['serial']}, O={atoms[i+3]['serial']}")
                
                # Skip past this backbone
                i += 4
            else:
                i += 1
        
        if backbone_groups:
            log_message(f"Found {len(backbone_groups)} backbone groups using sequential search")
            return backbone_groups
        
        # If sequential search failed, try distance-based approach
        log_message("Sequential search found no backbones. Trying distance-based search...")
        
        # Build distance-based bond connectivity
        def get_distance(atom1, atom2):
            pos1 = np.array([atom1['x'], atom1['y'], atom1['z']])
            pos2 = np.array([atom2['x'], atom2['y'], atom2['z']])
            return np.linalg.norm(pos1 - pos2)
        
        def are_bonded(atom1, atom2, tolerance=0.4):
            """Check if two atoms are bonded based on covalent radii"""
            covalent_radii = {
                'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05,
                'P': 1.07, 'F': 0.57, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39
            }
            r1 = covalent_radii.get(atom1['element'], 0.75)
            r2 = covalent_radii.get(atom2['element'], 0.75)
            expected_bond = r1 + r2
            distance = get_distance(atom1, atom2)
            return distance <= (expected_bond + tolerance)
        
        # Build adjacency list for all atoms
        n_atoms = len(atoms)
        bonds = {i: [] for i in range(n_atoms)}
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if are_bonded(atoms[i], atoms[j]):
                    bonds[i].append(j) 
                    bonds[j].append(i)
        
        # Find all N-C-C-O patterns using bonds
        # Find all N atoms
        n_indices = [i for i, atom in enumerate(atoms) if atom['element'] == 'N']
        
        for n_idx in n_indices:
            # Find C atoms bonded to N (potential CA)
            ca_candidates = [j for j in bonds[n_idx] if atoms[j]['element'] == 'C']
            
            for ca_idx in ca_candidates:
                # Find C atoms bonded to CA (potential C)
                c_candidates = [j for j in bonds[ca_idx] 
                               if atoms[j]['element'] == 'C' and j != n_idx]
                
                for c_idx in c_candidates:
                    # Find O atoms bonded to C
                    o_candidates = [j for j in bonds[c_idx] 
                                   if atoms[j]['element'] == 'O']
                    
                    if o_candidates:
                        # Found N-C-C-O pattern
                        o_idx = o_candidates[0]  # Take first O
                        
                        # Calculate average position for sorting
                        avg_pos = (atoms[n_idx]['serial'] + atoms[ca_idx]['serial'] + 
                                 atoms[c_idx]['serial'] + atoms[o_idx]['serial']) / 4
                        
                        backbone_groups.append({
                            'N': n_idx,
                            'CA': ca_idx,
                            'C': c_idx,
                            'O': o_idx,
                            'position': avg_pos
                        })
                        break  # Found backbone for this N, move to next
                if backbone_groups and backbone_groups[-1]['N'] == n_idx:
                    break  # Already found backbone for this N
        
        # If still no NCCO found, use fallback: find repeating N-C...C-O patterns
        if not backbone_groups:
            log_message("Distance-based search found no backbones. Using fallback pattern detection...")
            backbone_groups = self._find_backbone_fallback(atoms, bonds)
        
        # Sort by position
        backbone_groups.sort(key=lambda g: g['position'])
        
        log_message(f"Found {len(backbone_groups)} backbone groups")
        return backbone_groups
    
    def _find_backbone_fallback(self, atoms, bonds):
        """
        Fallback method: Find N atoms bonded to C, then find C bonded to O.
        Look for repeating patterns.
        """
        import numpy as np
        
        patterns = []
        n_indices = [i for i, atom in enumerate(atoms) if atom['element'] == 'N']
        
        for n_idx in n_indices:
            # Find all carbons bonded to N
            c_from_n = [j for j in bonds[n_idx] if atoms[j]['element'] == 'C']
            
            for c1_idx in c_from_n:
                # Find carbons bonded to this carbon
                c_from_c1 = [j for j in bonds[c1_idx] 
                            if atoms[j]['element'] == 'C' and j != n_idx]
                
                for c2_idx in c_from_c1:
                    # Find oxygen bonded to this second carbon
                    o_candidates = [j for j in bonds[c2_idx] 
                                   if atoms[j]['element'] == 'O']
                    
                    if o_candidates:
                        o_idx = o_candidates[0]
                        avg_pos = (atoms[n_idx]['serial'] + atoms[c1_idx]['serial'] + 
                                 atoms[c2_idx]['serial'] + atoms[o_idx]['serial']) / 4
                        
                        patterns.append({
                            'N': n_idx,
                            'CA': c1_idx,
                            'C': c2_idx,
                            'O': o_idx,
                            'position': avg_pos
                        })
                        break
                if patterns and patterns[-1]['N'] == n_idx:
                    break
        
        return patterns
    
    def _assign_atoms_to_residues(self, atoms, backbone_groups):
        """
        Assign atoms to residues based on backbone groups.
        Atoms between one NCCO and the next (or metal/strange atom) belong to that residue.
        """
        import numpy as np
        
        log_message("Assigning atoms to residues...")
        
        # Organic elements (everything else is "strange")
        ORGANIC_ELEMENTS = {'C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'}
        
        # Mark all backbone atoms
        backbone_indices = set()
        for group in backbone_groups:
            backbone_indices.update([group['N'], group['CA'], group['C'], group['O']])
        
        # Sort atoms by serial number to maintain order
        atom_order = sorted(range(len(atoms)), key=lambda i: atoms[i]['serial'])
        
        # Assign residue numbers
        residue_assignments = {}  # atom_idx -> residue_number
        current_res_num = 1
        
        for bb_idx, group in enumerate(backbone_groups):
            # Get the serial numbers of this backbone group
            bb_start = min(atoms[group['N']]['serial'], atoms[group['CA']]['serial'],
                          atoms[group['C']]['serial'], atoms[group['O']]['serial'])
            
            # Determine the end boundary
            if bb_idx < len(backbone_groups) - 1:
                next_group = backbone_groups[bb_idx + 1]
                bb_end = min(atoms[next_group['N']]['serial'], 
                           atoms[next_group['CA']]['serial'],
                           atoms[next_group['C']]['serial'], 
                           atoms[next_group['O']]['serial']) - 1
            else:
                bb_end = max(atom['serial'] for atom in atoms)
            
            # Assign backbone atoms to this residue
            for key in ['N', 'CA', 'C', 'O']:
                residue_assignments[group[key]] = current_res_num
                atoms[group[key]]['is_backbone'] = True
            
            # Find atoms between this backbone and next (or end)
            sidechain_atoms = []
            strange_atom_found = False
            strange_atom_serial = None
            
            for atom_idx in atom_order:
                atom_serial = atoms[atom_idx]['serial']
                
                # Skip if already assigned or outside range
                if atom_idx in residue_assignments:
                    continue
                if atom_serial < bb_start or atom_serial > bb_end:
                    continue
                
                # Check if strange atom (metal or non-organic)
                if atoms[atom_idx]['element'] not in ORGANIC_ELEMENTS:
                    strange_atom_found = True
                    strange_atom_serial = atom_serial
                    log_message(f"  Found strange atom {atoms[atom_idx]['element']} at serial {atom_serial}")
                    break
                
                sidechain_atoms.append(atom_idx)
            
            # Assign sidechain atoms to current residue
            for atom_idx in sidechain_atoms:
                # Stop if we've reached a strange atom
                if strange_atom_found and atoms[atom_idx]['serial'] >= strange_atom_serial:
                    break
                residue_assignments[atom_idx] = current_res_num
                atoms[atom_idx]['is_backbone'] = False
            
            # If strange atom found, assign remaining atoms to LIG residues
            if strange_atom_found:
                current_res_num += 1  # Move to next residue for the LIG
                lig_atoms = []
                
                for atom_idx in atom_order:
                    atom_serial = atoms[atom_idx]['serial']
                    
                    if atom_idx in residue_assignments:
                        continue
                    if atom_serial < strange_atom_serial or atom_serial > bb_end:
                        continue
                    
                    lig_atoms.append(atom_idx)
                    residue_assignments[atom_idx] = current_res_num
                    atoms[atom_idx]['is_backbone'] = False
                
                log_message(f"  Assigned {len(lig_atoms)} atoms to LIG residue {current_res_num}")
                current_res_num += 1  # Next regular residue
            else:
                current_res_num += 1
        
        # Group atoms by residue for identification
        residues = {}
        for atom_idx, res_num in residue_assignments.items():
            if res_num not in residues:
                residues[res_num] = []
            residues[res_num].append(atom_idx)
        
        # Identify amino acid type for each residue
        log_message("Identifying amino acid types...")
        for res_num, atom_indices in sorted(residues.items()):
            res_atoms = [atoms[i] for i in atom_indices]
            res_type = self._identify_amino_acid_type(res_atoms, atoms, atom_indices)
            
            log_message(f"  Residue {res_num}: {res_type} ({len(atom_indices)} atoms)")
            
            # Assign residue name to all atoms
            for atom_idx in atom_indices:
                atoms[atom_idx]['res_name'] = res_type
                atoms[atom_idx]['res_num'] = res_num
        
        return residues
    
    def _identify_amino_acid_type(self, res_atoms, all_atoms, atom_indices):
        """
        Identify amino acid type based on sidechain heavy atom composition.
        Uses element counts and topology for disambiguation.
        """
        # Build distance-based bonds within this residue
        def get_distance(idx1, idx2):
            import numpy as np
            pos1 = np.array([all_atoms[idx1]['x'], all_atoms[idx1]['y'], all_atoms[idx1]['z']])
            pos2 = np.array([all_atoms[idx2]['x'], all_atoms[idx2]['y'], all_atoms[idx2]['z']])
            return np.linalg.norm(pos1 - pos2)
        
        def are_bonded(idx1, idx2, tolerance=0.4):
            covalent_radii = {
                'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05,
                'P': 1.07, 'F': 0.57, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39
            }
            r1 = covalent_radii.get(all_atoms[idx1]['element'], 0.75)
            r2 = covalent_radii.get(all_atoms[idx2]['element'], 0.75)
            expected_bond = r1 + r2
            distance = get_distance(idx1, idx2)
            return distance <= (expected_bond + tolerance)
        
        # Get sidechain heavy atoms (non-backbone, non-hydrogen)
        backbone_atoms = {'N', 'CA', 'C', 'O'}
        sidechain_indices = []
        
        for idx in atom_indices:
            atom = all_atoms[idx]
            # Skip hydrogens
            if atom['element'] == 'H':
                continue
            # Skip backbone atoms
            if atom.get('is_backbone', False):
                continue
            sidechain_indices.append(idx)
        
        # Count elements in sidechain
        element_counts = {}
        for idx in sidechain_indices:
            elem = all_atoms[idx]['element']
            element_counts[elem] = element_counts.get(elem, 0) + 1
        
        n_C = element_counts.get('C', 0)
        n_N = element_counts.get('N', 0)
        n_O = element_counts.get('O', 0)
        n_S = element_counts.get('S', 0)
        
        # Amino acid identification logic matching the counts I listed
        
        # GLY: 0 sidechain atoms
        if len(sidechain_indices) == 0:
            return 'GLY'
        
        # ALA: 1 C (CB)
        if n_C == 1 and n_N == 0 and n_O == 0 and n_S == 0:
            return 'ALA'
        
        # SER: 1 C, 1 O (CB, OG)
        if n_C == 1 and n_O == 1 and n_N == 0 and n_S == 0:
            return 'SER'
        
        # CYS: 1 C, 1 S (CB, SG)
        if n_C == 1 and n_S == 1 and n_N == 0 and n_O == 0:
            return 'CYS'
        
        # THR: 2 C, 1 O (CB, OG1, CG2)
        if n_C == 2 and n_O == 1 and n_N == 0 and n_S == 0:
            return 'THR'
        
        # ASN: 2 C, 1 O, 1 N (CB, CG, OD1, ND2)
        if n_C == 2 and n_O == 1 and n_N == 1 and n_S == 0:
            return 'ASN'
        
        # ASP: 2 C, 2 O (CB, CG, OD1, OD2)
        if n_C == 2 and n_O == 2 and n_N == 0 and n_S == 0:
            return 'ASP'
        
        # VAL vs PRO: both have 3 C
        # PRO: 3 C where CD bonds back to backbone N (forms ring)
        # VAL: 3 C where CB bonds to CA, CG1, CG2 (CB has 3 C neighbors)
        if n_C == 3 and n_N == 0 and n_O == 0 and n_S == 0:
            # Find CA index (backbone carbon)
            ca_idx = None
            for idx in atom_indices:
                if all_atoms[idx].get('is_backbone', False) and all_atoms[idx]['element'] == 'C':
                    # CA is the backbone carbon bonded to N
                    ca_idx = idx
                    break
            
            if ca_idx is not None:
                # Check each sidechain carbon
                for sc_idx in sidechain_indices:
                    # Count how many carbons this sidechain carbon bonds to (including CA)
                    c_neighbors = 0
                    for other_idx in atom_indices:
                        if all_atoms[other_idx]['element'] == 'C' and other_idx != sc_idx:
                            if are_bonded(sc_idx, other_idx):
                                c_neighbors += 1
                    
                    # If a sidechain carbon bonds to 3 other carbons, it's VAL
                    if c_neighbors == 3:
                        return 'VAL'
            
            # If no carbon with 3 carbon neighbors, it's PRO
            return 'PRO'
        
        # GLN: 3 C, 1 O, 1 N (CB, CG, CD, OE1, NE2)
        if n_C == 3 and n_O == 1 and n_N == 1 and n_S == 0:
            return 'GLN'
        
        # GLU: 3 C, 2 O (CB, CG, CD, OE1, OE2)
        if n_C == 3 and n_O == 2 and n_N == 0 and n_S == 0:
            return 'GLU'
        
        # ILE vs LEU: both have 4 C
        # ILE: CB bonds to CA, CG1, CG2; CG1 bonds to CD1
        #      (The C bonded to 3 other C's is CB, which is bonded to CA from NCCO)
        # LEU: CB bonds to CA, CG; CG bonds to CD1, CD2
        #      (The C bonded to 3 other C's is CG, which is NOT bonded to CA from NCCO)
        if n_C == 4 and n_N == 0 and n_O == 0 and n_S == 0:
            # Find CA index
            ca_idx = None
            for idx in atom_indices:
                if all_atoms[idx].get('is_backbone', False) and all_atoms[idx]['element'] == 'C':
                    ca_idx = idx
                    break
            
            if ca_idx is not None:
                # Check each sidechain carbon
                for sc_idx in sidechain_indices:
                    # Count carbon neighbors
                    c_neighbors = []
                    for other_idx in atom_indices:
                        if all_atoms[other_idx]['element'] == 'C' and other_idx != sc_idx:
                            if are_bonded(sc_idx, other_idx):
                                c_neighbors.append(other_idx)
                    
                    # If this carbon has 3 carbon neighbors
                    if len(c_neighbors) == 3:
                        # Check if one of them is CA (backbone)
                        if ca_idx in c_neighbors:
                            return 'ILE'  # CB bonded to CA
                        else:
                            return 'LEU'  # CG bonded to CB (not CA)
            
            return 'LEU'  # Default to LEU if unclear
        
        # MET: 3 C, 1 S (CB, CG, SD, CE)
        if n_C == 3 and n_S == 1 and n_N == 0 and n_O == 0:
            return 'MET'
        
        # LYS: 4 C, 1 N (CB, CG, CD, CE, NZ)
        if n_C == 4 and n_N == 1 and n_O == 0 and n_S == 0:
            return 'LYS'
        
        # ARG: 4 C, 3 N (CB, CG, CD, NE, CZ, NH1, NH2)
        if n_C == 4 and n_N == 3 and n_O == 0 and n_S == 0:
            return 'ARG'
        
        # HIS: 4 C, 2 N (CB, CG, ND1, CD2, CE1, NE2)
        if n_C == 4 and n_N == 2 and n_O == 0 and n_S == 0:
            return 'HIS'
        
        # PHE: 7 C (CB, CG, CD1, CD2, CE1, CE2, CZ)
        if n_C == 7 and n_N == 0 and n_O == 0 and n_S == 0:
            return 'PHE'
        
        # TYR: 7 C, 1 O (CB, CG, CD1, CD2, CE1, CE2, CZ, OH)
        if n_C == 7 and n_O == 1 and n_N == 0 and n_S == 0:
            return 'TYR'
        
        # TRP: 9 C, 1 N (CB, CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2)
        if n_C == 9 and n_N == 1 and n_O == 0 and n_S == 0:
            return 'TRP'
        
        # Check if sidechain contains only organic elements
        ORGANIC_ELEMENTS = {'C', 'H', 'N', 'O', 'S', 'P'}
        all_organic = all(all_atoms[idx]['element'] in ORGANIC_ELEMENTS 
                         for idx in atom_indices)
        
        if all_organic:
            log_message(f"    Unmatched organic sidechain: C={n_C}, N={n_N}, O={n_O}, S={n_S}")
            return 'LIG'  # Organic but not standard amino acid
        else:
            return 'LIG'  # Contains non-organic atoms
    
    def _write_identified_pdb(self, atoms, output_pdb):
        """
        Write atoms to PDB file with proper formatting and correct residue information.
        Maintains original atom order (including hydrogens) and assigns atom names.
        """
        log_message(f"Writing identified PDB to {output_pdb}...")
        
        # First, assign proper atom names within each residue
        self._assign_standard_atom_names(atoms)
        
        # Order residues and detect terminal caps
        self._order_residues_and_detect_caps(atoms)
        
        # atoms are already sorted by _renumber_residues_by_chain
        
        with open(output_pdb, 'w') as f:
            for atom in atoms:
                # Format PDB line
                line = self._format_pdb_atom_line(atom)
                f.write(line + '\n')
            f.write('END\n')
        
        log_message(f"Successfully wrote {len(atoms)} atoms to {output_pdb}")
    
    def _assign_standard_atom_names(self, atoms):
        """
        Assign standard PDB atom names to all atoms in each residue.
        For standard amino acids, use standard nomenclature.
        For LIG residues, use topology-based Greek letter naming.
        """
        # Standard amino acid atom naming templates
        STANDARD_AA_ATOMS = {
            'GLY': {'N', 'CA', 'C', 'O'},
            'ALA': {'N', 'CA', 'C', 'O', 'CB'},
            'SER': {'N', 'CA', 'C', 'O', 'CB', 'OG'},
            'CYS': {'N', 'CA', 'C', 'O', 'CB', 'SG'},
            'VAL': {'N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'},
            'THR': {'N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'},
            'ILE': {'N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'},
            'LEU': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'},
            'ASN': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'},
            'ASP': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'},
            'GLN': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'},
            'GLU': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'},
            'MET': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'},
            'LYS': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'},
            'ARG': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'},
            'HIS': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'},
            'PHE': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'},
            'TYR': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'},
            'TRP': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'},
            'PRO': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'}
        }
        
        # Group atoms by residue
        residues = {}
        for atom in atoms:
            res_num = atom.get('res_num', 1)
            if res_num not in residues:
                residues[res_num] = []
            residues[res_num].append(atom)
        
        # Process each residue
        for res_num, res_atoms in residues.items():
            res_type = res_atoms[0]['res_name']
            
            if res_type in STANDARD_AA_ATOMS:
                self._assign_standard_aa_names(res_atoms, res_type, STANDARD_AA_ATOMS[res_type])
            else:  # LIG or non-standard
                self._assign_topology_based_names(res_atoms)
    
    def _assign_standard_aa_names(self, res_atoms, res_type, standard_atoms):
        """Assign standard PDB atom names to a standard amino acid residue"""
        import numpy as np
        
        # Build bonds within residue
        def get_distance(atom1, atom2):
            pos1 = np.array([atom1['x'], atom1['y'], atom1['z']])
            pos2 = np.array([atom2['x'], atom2['y'], atom2['z']])
            return np.linalg.norm(pos1 - pos2)
        
        def are_bonded(atom1, atom2, tolerance=0.4):
            covalent_radii = {
                'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05,
                'P': 1.07
            }
            r1 = covalent_radii.get(atom1['element'], 0.75)
            r2 = covalent_radii.get(atom2['element'], 0.75)
            expected_bond = r1 + r2
            distance = get_distance(atom1, atom2)
            return distance <= (expected_bond + tolerance)
        
        # Separate heavy atoms and hydrogens
        heavy_atoms = [a for a in res_atoms if a['element'] != 'H']
        hydrogens = [a for a in res_atoms if a['element'] == 'H']
        
        # Build bond graph for heavy atoms
        n_heavy = len(heavy_atoms)
        bonds = {i: [] for i in range(n_heavy)}
        for i in range(n_heavy):
            for j in range(i + 1, n_heavy):
                if are_bonded(heavy_atoms[i], heavy_atoms[j]):
                    bonds[i].append(j)
                    bonds[j].append(i)
        
        # Assign names based on residue type and topology
        # First, find backbone atoms
        n_idx = None
        ca_idx = None
        c_idx = None
        o_idx = None
        
        for i, atom in enumerate(heavy_atoms):
            if atom.get('is_backbone', False):
                if atom['element'] == 'N':
                    n_idx = i
                    atom['atom_name'] = 'N'
                elif atom['element'] == 'C':
                    # Distinguish between CA and C
                    # CA bonds to more atoms typically
                    if ca_idx is None:
                        ca_idx = i
                        atom['atom_name'] = 'CA'
                    else:
                        c_idx = i
                        atom['atom_name'] = 'C'
                elif atom['element'] == 'O':
                    o_idx = i
                    atom['atom_name'] = 'O'
        
        # If we have two carbons in backbone, ensure CA is the one bonded to N
        if n_idx is not None and ca_idx is not None and c_idx is not None:
            if c_idx in bonds[n_idx] and ca_idx not in bonds[n_idx]:
                # Swap CA and C assignments
                heavy_atoms[ca_idx]['atom_name'] = 'C'
                heavy_atoms[c_idx]['atom_name'] = 'CA'
                ca_idx, c_idx = c_idx, ca_idx
        
        # Now assign sidechain atoms using standard naming
        self._assign_sidechain_standard_names(heavy_atoms, bonds, res_type, ca_idx)
        
        # Assign hydrogen names based on parent heavy atom
        self._assign_hydrogen_names(heavy_atoms, hydrogens)
    
    def _assign_sidechain_standard_names(self, heavy_atoms, bonds, res_type, ca_idx):
        """Assign standard sidechain atom names based on topology"""
        import numpy as np
        from collections import deque
        
        if ca_idx is None:
            return
        
        # BFS from CA to assign names by distance
        visited = set([ca_idx])
        queue = deque([ca_idx])
        level = {ca_idx: 0}
        
        while queue:
            curr_idx = queue.popleft()
            for neighbor_idx in bonds[curr_idx]:
                if neighbor_idx not in visited and not heavy_atoms[neighbor_idx].get('is_backbone', False):
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)
                    level[neighbor_idx] = level[curr_idx] + 1
        
        # Group sidechain atoms by level and element
        sidechain_by_level = {}
        for idx, lv in level.items():
            if lv > 0 and not heavy_atoms[idx].get('is_backbone', False):
                if lv not in sidechain_by_level:
                    sidechain_by_level[lv] = []
                sidechain_by_level[lv].append(idx)
        
        # Use Greek letters: B, G, D, E, Z
        greek_letters = ['B', 'G', 'D', 'E', 'Z', 'H']
        
        for lv in sorted(sidechain_by_level.keys()):
            if lv - 1 >= len(greek_letters):
                continue
            
            letter = greek_letters[lv - 1]
            atoms_at_level = sidechain_by_level[lv]
            
            # Sort by element (consistent ordering)
            atoms_at_level.sort(key=lambda idx: (heavy_atoms[idx]['element'], idx))
            
            for i, idx in enumerate(atoms_at_level):
                elem = heavy_atoms[idx]['element']
                if len(atoms_at_level) == 1:
                    heavy_atoms[idx]['atom_name'] = f"{elem}{letter}"
                else:
                    heavy_atoms[idx]['atom_name'] = f"{elem}{letter}{i+1}"
    
    def _assign_topology_based_names(self, res_atoms):
        """Assign topology-based Greek letter names to LIG residues.
        For LIG residues containing metals, use sequential numbering (Ru0, C1, H2, etc.).
        For organic-only LIG residues, use Greek letter naming."""
        import numpy as np
        from collections import deque
        
        # Check if this LIG contains a metal
        METAL_ELEMENTS = {
            'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 
            'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
            'In', 'Sn', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 
            'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 
            'Bi', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'
        }
        
        has_metal = any(atom['element'] in METAL_ELEMENTS for atom in res_atoms)
        
        if has_metal:
            # Use sequential numbering for metal-containing LIG
            self._assign_sequential_metal_names(res_atoms)
        else:
            # Use Greek letter naming for organic-only LIG
            self._assign_greek_letter_names(res_atoms)
    
    def _assign_sequential_metal_names(self, res_atoms):
        """Assign sequential numbering to metal-containing LIG residues (Ru0, C1, H2, etc.)"""
        import numpy as np
        
        # Build bonds
        def get_distance(atom1, atom2):
            pos1 = np.array([atom1['x'], atom1['y'], atom1['z']])
            pos2 = np.array([atom2['x'], atom2['y'], atom2['z']])
            return np.linalg.norm(pos1 - pos2)
        
        def are_bonded(atom1, atom2, tolerance=0.4):
            covalent_radii = {
                'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05,
                'P': 1.07, 'Fe': 1.32, 'Cu': 1.32, 'Zn': 1.31, 'Ru': 1.46,
                'Rh': 1.42, 'Pd': 1.39, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36
            }
            r1 = covalent_radii.get(atom1['element'], 1.5)
            r2 = covalent_radii.get(atom2['element'], 1.5)
            expected_bond = r1 + r2
            distance = get_distance(atom1, atom2)
            return distance <= (expected_bond + tolerance)
        
        # Sort atoms by serial number to maintain order
        sorted_atoms = sorted(res_atoms, key=lambda a: a['serial'])
        
        # Assign sequential names starting from 0
        for i, atom in enumerate(sorted_atoms):
            elem = atom['element']
            atom['atom_name'] = f"{elem}{i}"
        
        log_message(f"Assigned sequential metal naming to {len(sorted_atoms)} atoms")
    
    def _assign_greek_letter_names(self, res_atoms):
        """Assign Greek letter naming to organic-only LIG residues"""
        import numpy as np
        from collections import deque
        
        # Build bonds
        def get_distance(atom1, atom2):
            pos1 = np.array([atom1['x'], atom1['y'], atom1['z']])
            pos2 = np.array([atom2['x'], atom2['y'], atom2['z']])
            return np.linalg.norm(pos1 - pos2)
        
        def are_bonded(atom1, atom2, tolerance=0.4):
            covalent_radii = {
                'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05,
                'P': 1.07, 'Fe': 1.32, 'Cu': 1.32, 'Zn': 1.31
            }
            r1 = covalent_radii.get(atom1['element'], 0.75)
            r2 = covalent_radii.get(atom2['element'], 0.75)
            expected_bond = r1 + r2
            distance = get_distance(atom1, atom2)
            return distance <= (expected_bond + tolerance)
        
        # Separate heavy and hydrogens
        heavy_atoms = [a for a in res_atoms if a['element'] != 'H']
        hydrogens = [a for a in res_atoms if a['element'] == 'H']
        
        if not heavy_atoms:
            return
        
        # Build bond graph
        n_heavy = len(heavy_atoms)
        bonds = {i: [] for i in range(n_heavy)}
        for i in range(n_heavy):
            for j in range(i + 1, n_heavy):
                if are_bonded(heavy_atoms[i], heavy_atoms[j]):
                    bonds[i].append(j)
                    bonds[j].append(i)
        
        # Start from first heavy atom (by serial number)
        start_idx = 0
        
        # BFS to assign levels
        visited = set([start_idx])
        queue = deque([start_idx])
        level = {start_idx: 0}
        
        while queue:
            curr_idx = queue.popleft()
            for neighbor_idx in bonds[curr_idx]:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)
                    level[neighbor_idx] = level[curr_idx] + 1
        
        # Assign names using Greek letters
        greek_letters = ['A', 'B', 'G', 'D', 'E', 'Z', 'H', 'I', 'K', 'L', 'M', 'N']
        
        # Group by level
        by_level = {}
        for idx, lv in level.items():
            if lv not in by_level:
                by_level[lv] = []
            by_level[lv].append(idx)
        
        for lv in sorted(by_level.keys()):
            if lv >= len(greek_letters):
                letter = 'X'
            else:
                letter = greek_letters[lv]
            
            atoms_at_level = by_level[lv]
            atoms_at_level.sort(key=lambda idx: (heavy_atoms[idx]['element'], idx))
            
            for i, idx in enumerate(atoms_at_level):
                elem = heavy_atoms[idx]['element']
                if len(atoms_at_level) == 1:
                    heavy_atoms[idx]['atom_name'] = f"{elem}{letter}"
                else:
                    heavy_atoms[idx]['atom_name'] = f"{elem}{letter}{i+1}"
        
        # Assign hydrogen names
        self._assign_hydrogen_names(heavy_atoms, hydrogens)
    
    def _assign_hydrogen_names(self, heavy_atoms, hydrogens):
        """Assign hydrogen names based on parent heavy atom (1.2 Å cutoff)"""
        import numpy as np
        
        def get_distance(atom1, atom2):
            pos1 = np.array([atom1['x'], atom1['y'], atom1['z']])
            pos2 = np.array([atom2['x'], atom2['y'], atom2['z']])
            return np.linalg.norm(pos1 - pos2)
        
        # For each hydrogen, find parent heavy atom
        h_cutoff = 1.2
        
        # First pass: assign parent indices
        h_parents = {}  # hydrogen index -> parent heavy atom index
        for h_idx, h_atom in enumerate(hydrogens):
            parent_idx = None
            min_dist = h_cutoff
            
            for i, heavy in enumerate(heavy_atoms):
                dist = get_distance(h_atom, heavy)
                if dist < min_dist:
                    min_dist = dist
                    parent_idx = i
            
            if parent_idx is not None:
                h_parents[h_idx] = parent_idx
        
        # Second pass: count hydrogens per parent and assign names
        parent_h_counts = {}  # parent index -> count of assigned hydrogens
        
        for h_idx, h_atom in enumerate(hydrogens):
            parent_idx = h_parents.get(h_idx)
            
            if parent_idx is not None:
                parent_name = heavy_atoms[parent_idx].get('atom_name', 'X')
                
                # Get current count for this parent
                current_count = parent_h_counts.get(parent_idx, 0)
                
                # Generate hydrogen name
                if parent_name.startswith('N'):
                    # N-terminal or backbone N hydrogens
                    if current_count == 0:
                        h_atom['atom_name'] = 'H'
                    else:
                        h_atom['atom_name'] = f'H{current_count + 1}'
                elif parent_name.startswith('C') or parent_name.startswith('O') or parent_name.startswith('S'):
                    # Remove element letter, keep Greek letter
                    if len(parent_name) >= 2:
                        greek = parent_name[1]
                        h_atom['atom_name'] = f'H{greek}{current_count + 1}'
                    else:
                        h_atom['atom_name'] = f'H{current_count + 1}'
                else:
                    h_atom['atom_name'] = f'H{current_count + 1}'
                
                # Increment count
                parent_h_counts[parent_idx] = current_count + 1
            else:
                # No parent found, assign generic name
                h_atom['atom_name'] = 'H'

    def _order_residues_and_detect_caps(self, atoms):
        """
        Order residues based on peptide chain connectivity and detect/add terminal caps.
        This should be called after _assign_standard_atom_names but before _write_identified_pdb.
        """
        import numpy as np
        
        log_message("Ordering residues and detecting terminal caps...")
        
        # Build bond connectivity
        def get_distance(atom1, atom2):
            pos1 = np.array([atom1['x'], atom1['y'], atom1['z']])
            pos2 = np.array([atom2['x'], atom2['y'], atom2['z']])
            return np.linalg.norm(pos1 - pos2)
        
        def are_bonded(atom1, atom2, max_dist=1.6):
            """Check if two atoms are within bonding distance"""
            return get_distance(atom1, atom2) <= max_dist
        
        # Group atoms by residue
        residues = {}
        for atom in atoms:
            res_num = atom.get('res_num', 1)
            if res_num not in residues:
                residues[res_num] = []
            residues[res_num].append(atom)
        
        # Identify which residues have NCCO (are part of chain)
        chain_residues = set()
        non_chain_ligs = set()
        
        for res_num, res_atoms in residues.items():
            res_name = res_atoms[0]['res_name']
            
            # Check if residue has NCCO backbone
            has_ncco = any(atom.get('is_backbone', False) for atom in res_atoms)
            
            if has_ncco:
                chain_residues.add(res_num)
            elif res_name == 'LIG':
                non_chain_ligs.add(res_num)
        
        log_message(f"Found {len(chain_residues)} chain residues and {len(non_chain_ligs)} non-chain LIG residues")
        
        # Build chain order based on C-N connectivity
        chain_order = self._build_chain_order(atoms, residues, chain_residues)
        
        if not chain_order:
            log_message("Warning: Could not build chain order. Using original numbering.")
            return
        
        log_message(f"Chain order: {chain_order}")
        
        # Check N-terminal (first residue in chain)
        self._check_n_terminal(atoms, residues, chain_order[0])
        
        # Check and extract C-terminal cap
        terminal_res_num = chain_order[-1]
        cap_residue_num = self._detect_and_extract_c_terminal_cap(
            atoms, residues, terminal_res_num, non_chain_ligs, chain_residues
        )
        
        # If cap was found, add it to chain order
        if cap_residue_num is not None:
            chain_order.append(cap_residue_num)
            # Update residues dict with cap residue
            cap_atoms = [a for a in atoms if a.get('res_num') == cap_residue_num]
            residues[cap_residue_num] = cap_atoms
            log_message(f"Added C-terminal cap as residue {cap_residue_num} to chain order")
        
        # Position non-chain LIGs with metals near their coordination residues
        positioned_ligs = self._position_metal_ligs(atoms, residues, chain_order, non_chain_ligs)
        
        # Renumber residues following chain order, positioned LIGs, then remaining LIGs, then cap
        self._renumber_and_reorder_residues(atoms, chain_order, positioned_ligs, non_chain_ligs, cap_residue_num)
    
    def _build_chain_order(self, atoms, residues, chain_residues):
        """Build the order of residues in the peptide chain based on C-N connectivity"""
        import numpy as np
        
        log_message("Building peptide chain order...")
        
        def get_distance(atom1, atom2):
            pos1 = np.array([atom1['x'], atom1['y'], atom1['z']])
            pos2 = np.array([atom2['x'], atom2['y'], atom2['z']])
            return np.linalg.norm(pos1 - pos2)
        
        # Find C and N atoms for each chain residue
        residue_terminals = {}  # res_num -> {'C': atom, 'N': atom}
        
        for res_num in chain_residues:
            res_atoms = residues[res_num]
            
            # Find backbone C and N
            c_atom = None
            n_atom = None
            
            for atom in res_atoms:
                if atom.get('is_backbone', False):
                    if atom['element'] == 'C' and atom.get('atom_name') == 'C':
                        c_atom = atom
                    elif atom['element'] == 'N' and atom.get('atom_name') == 'N':
                        n_atom = atom
            
            if c_atom and n_atom:
                residue_terminals[res_num] = {'C': c_atom, 'N': n_atom}
        
        # Build connectivity graph: which residue's C connects to which residue's N
        peptide_bonds = {}  # res_num -> next_res_num
        reverse_bonds = {}  # res_num -> prev_res_num
        
        for res1_num, term1 in residue_terminals.items():
            c_atom = term1['C']
            
            for res2_num, term2 in residue_terminals.items():
                if res1_num == res2_num:
                    continue
                
                n_atom = term2['N']
                
                # Check if C of res1 is close to N of res2 (peptide bond ~1.33 Å)
                dist = get_distance(c_atom, n_atom)
                if dist < 1.6:  # Generous cutoff for peptide bond
                    peptide_bonds[res1_num] = res2_num
                    reverse_bonds[res2_num] = res1_num
                    log_message(f"  Found peptide bond: residue {res1_num} -> {res2_num} (distance: {dist:.2f} Å)")
        
        # Find the start of the chain (residue with N but no preceding C)
        start_residue = None
        for res_num in chain_residues:
            if res_num not in reverse_bonds:
                start_residue = res_num
                break
        
        if start_residue is None:
            log_message("Warning: Could not find chain start. Using lowest residue number.")
            start_residue = min(chain_residues)
        
        # Build chain by following peptide bonds
        chain_order = [start_residue]
        current = start_residue
        
        while current in peptide_bonds:
            next_res = peptide_bonds[current]
            if next_res in chain_order:  # Prevent cycles
                break
            chain_order.append(next_res)
            current = next_res
        
        # Add any remaining chain residues that weren't connected
        for res_num in chain_residues:
            if res_num not in chain_order:
                log_message(f"Warning: Residue {res_num} not connected to main chain. Adding at end.")
                chain_order.append(res_num)
        
        return chain_order
    
    def _check_n_terminal(self, atoms, residues, first_res_num):
        """Check N-terminal residue and rename first hydrogen from H to H1"""
        log_message(f"Checking N-terminal residue {first_res_num}...")
        
        res_atoms = residues[first_res_num]
        
        # Find backbone N
        n_atom = None
        for atom in res_atoms:
            if atom.get('is_backbone', False) and atom['element'] == 'N':
                n_atom = atom
                break
        
        if not n_atom:
            log_message("Warning: Could not find backbone N in first residue")
            return
        
        # Find hydrogens bonded to N
        n_hydrogens = []
        for atom in res_atoms:
            if atom['element'] == 'H':
                import numpy as np
                pos_n = np.array([n_atom['x'], n_atom['y'], n_atom['z']])
                pos_h = np.array([atom['x'], atom['y'], atom['z']])
                dist = np.linalg.norm(pos_n - pos_h)
                
                if dist < 1.2:  # H-N bond length
                    n_hydrogens.append(atom)
        
        if len(n_hydrogens) > 1:
            log_message(f"N-terminal detected: Found {len(n_hydrogens)} hydrogens on backbone N")
        else:
            log_message(f"N-terminal check: Found {len(n_hydrogens)} hydrogen(s) on backbone N")
        
        # Rename hydrogens: H -> H1, H2, H3
        for i, h_atom in enumerate(n_hydrogens):
            old_name = h_atom.get('atom_name', 'H')
            new_name = f'H{i+1}'
            h_atom['atom_name'] = new_name
            if old_name != new_name:
                log_message(f"  Renamed N-terminal hydrogen: {old_name} -> {new_name}")
    
    def _detect_and_extract_c_terminal_cap(self, atoms, residues, terminal_res_num, non_chain_ligs, chain_residues):
        """Detect C-terminal NH2 cap and extract it as a separate residue"""
        import numpy as np
        
        log_message(f"Checking for C-terminal cap on residue {terminal_res_num}...")
        
        terminal_atoms = residues[terminal_res_num]
        terminal_res_name = terminal_atoms[0]['res_name']
        
        def get_distance(atom1, atom2):
            pos1 = np.array([atom1['x'], atom1['y'], atom1['z']])
            pos2 = np.array([atom2['x'], atom2['y'], atom2['z']])
            return np.linalg.norm(pos1 - pos2)
        
        # Find terminal C (carbonyl carbon)
        terminal_c = None
        for atom in terminal_atoms:
            if atom.get('is_backbone', False) and atom['element'] == 'C' and atom.get('atom_name') == 'C':
                terminal_c = atom
                break
        
        if not terminal_c:
            log_message("Warning: Could not find terminal C atom")
            return None
        
        cap_n = None
        cap_hydrogens = []
        cap_source_lig = None
        
        # Case 1: Terminal residue is LIG with NCCO
        if terminal_res_name == 'LIG':
            log_message("Terminal residue is LIG. Checking for bonded N...")
            
            # Look for N bonded to terminal C within this LIG
            for atom in terminal_atoms:
                if atom['element'] == 'N' and not atom.get('is_backbone', False):
                    dist = get_distance(terminal_c, atom)
                    if dist < 1.6:  # C-N bond
                        cap_n = atom
                        log_message(f"  Found N bonded to terminal C (distance: {dist:.2f} Å)")
                        break
            
            if cap_n:
                # Find hydrogens bonded to this N
                for atom in terminal_atoms:
                    if atom['element'] == 'H':
                        dist = get_distance(cap_n, atom)
                        if dist < 1.2:  # N-H bond
                            cap_hydrogens.append(atom)
                
                log_message(f"  Found {len(cap_hydrogens)} hydrogens bonded to cap N")
                
                # Try to identify residue without the cap
                remaining_atoms = [a for a in terminal_atoms 
                                 if a != cap_n and a not in cap_hydrogens]
                
                # Re-identify without cap atoms
                remaining_indices = [atoms.index(a) for a in remaining_atoms]
                identified_type = self._identify_amino_acid_type(
                    remaining_atoms, atoms, remaining_indices
                )
                
                if identified_type != 'LIG':
                    log_message(f"  Terminal LIG identified as {identified_type} without cap")
                    # Rename terminal residue
                    for atom in remaining_atoms:
                        atom['res_name'] = identified_type
                else:
                    log_message("  Terminal LIG remains LIG without cap")
                
                cap_source_lig = terminal_res_num
        
        # Case 2: Terminal is not LIG, search non-chain LIGs for cap N
        else:
            log_message("Terminal residue is not LIG. Searching non-chain LIG residues for cap N...")
            
            for lig_num in non_chain_ligs:
                lig_atoms = residues[lig_num]
                
                for atom in lig_atoms:
                    if atom['element'] == 'N':
                        dist = get_distance(terminal_c, atom)
                        if dist < 1.6:  # C-N bond
                            cap_n = atom
                            cap_source_lig = lig_num
                            log_message(f"  Found cap N in LIG {lig_num} (distance: {dist:.2f} Å)")
                            break
                
                if cap_n:
                    # Find hydrogens bonded to this N
                    for atom in lig_atoms:
                        if atom['element'] == 'H':
                            dist = get_distance(cap_n, atom)
                            if dist < 1.2:
                                cap_hydrogens.append(atom)
                    
                    log_message(f"  Found {len(cap_hydrogens)} hydrogens bonded to cap N")
                    break
        
        # If cap N found, extract it as separate residue
        if cap_n:
            cap_atoms = [cap_n] + cap_hydrogens
            
            # Find a new residue number that doesn't conflict
            max_res_num = max(atom.get('res_num', 1) for atom in atoms)
            new_cap_res_num = max_res_num + 1
            
            # Update cap atoms to be their own residue
            for atom in cap_atoms:
                atom['res_name'] = 'NH2'
                atom['res_num'] = new_cap_res_num
            
            # Rename N and hydrogens
            cap_n['atom_name'] = 'N'
            for i, h_atom in enumerate(cap_hydrogens):
                h_atom['atom_name'] = f'H{i+1}'
            
            # Remove cap from source LIG if it came from a LIG
            if cap_source_lig in non_chain_ligs:
                non_chain_ligs.remove(cap_source_lig)
                # Check if source LIG still has atoms
                remaining_lig_atoms = [a for a in residues[cap_source_lig] if a not in cap_atoms]
                if remaining_lig_atoms:
                    # LIG still has atoms, keep it
                    non_chain_ligs.add(cap_source_lig)
                    residues[cap_source_lig] = remaining_lig_atoms
                else:
                    # LIG is now empty, remove it
                    del residues[cap_source_lig]
                    log_message(f"  LIG {cap_source_lig} was only the cap, removed from non-chain LIGs")
            
            log_message(f"Extracted C-terminal cap NH2 as separate residue {new_cap_res_num} with {len(cap_atoms)} atoms")
            
            return new_cap_res_num
        
        log_message("No C-terminal cap detected")
        return None
    
    def _position_metal_ligs(self, atoms, residues, chain_order, non_chain_ligs):
        """
        For each non-chain LIG containing metal, find the closest chain atom
        and determine where to position the LIG relative to chain residues.
        Returns dict: {lig_num: (insert_after_res_num, coordination_distance)}
        """
        import numpy as np
        
        METAL_ELEMENTS = {
            'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 
            'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
            'In', 'Sn', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 
            'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 
            'Bi', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'
        }
        
        def get_distance(atom1, atom2):
            pos1 = np.array([atom1['x'], atom1['y'], atom1['z']])
            pos2 = np.array([atom2['x'], atom2['y'], atom2['z']])
            return np.linalg.norm(pos1 - pos2)
        
        positioned_ligs = {}  # lig_num -> (after_res_num, distance)
        
        # Get all chain atoms (from both regular and LIG chain residues)
        chain_atoms = []
        for res_num in chain_order:
            chain_atoms.extend(residues[res_num])
        
        log_message(f"Positioning {len(non_chain_ligs)} non-chain LIG residues...")
        
        for lig_num in non_chain_ligs:
            lig_atoms = residues[lig_num]
            
            # Check if this LIG contains a metal
            has_metal = any(atom['element'] in METAL_ELEMENTS for atom in lig_atoms)
            
            if not has_metal:
                log_message(f"  LIG {lig_num} has no metal, will place at end")
                continue
            
            # Find metal atoms in this LIG
            metal_atoms = [a for a in lig_atoms if a['element'] in METAL_ELEMENTS]
            
            log_message(f"  LIG {lig_num} contains {len(metal_atoms)} metal atom(s)")
            
            # Find closest chain atom to any metal in this LIG
            min_distance = float('inf')
            closest_chain_atom = None
            closest_metal = None
            
            for metal_atom in metal_atoms:
                for chain_atom in chain_atoms:
                    dist = get_distance(metal_atom, chain_atom)
                    if dist < min_distance:
                        min_distance = dist
                        closest_chain_atom = chain_atom
                        closest_metal = metal_atom
            
            if closest_chain_atom:
                # Find which residue this chain atom belongs to
                coord_res_num = closest_chain_atom['res_num']
                
                log_message(f"  LIG {lig_num} metal {closest_metal['element']} closest to "
                          f"residue {coord_res_num} atom {closest_chain_atom.get('atom_name', 'UNK')} "
                          f"(distance: {min_distance:.2f} Å)")
                
                # If coordinated to multiple residues (find all within coordination distance)
                coordination_cutoff = min_distance + 0.5  # Generous cutoff
                coordinated_residues = set()
                
                for metal_atom in metal_atoms:
                    for chain_atom in chain_atoms:
                        dist = get_distance(metal_atom, chain_atom)
                        if dist < coordination_cutoff:
                            coordinated_residues.add(chain_atom['res_num'])
                
                if len(coordinated_residues) > 1:
                    log_message(f"  LIG {lig_num} coordinated to multiple residues: {sorted(coordinated_residues)}")
                    # Use the first one in chain order
                    for res_num in chain_order:
                        if res_num in coordinated_residues:
                            coord_res_num = res_num
                            log_message(f"  Using first coordinated residue in chain: {coord_res_num}")
                            break
                
                positioned_ligs[lig_num] = (coord_res_num, min_distance)
        
        return positioned_ligs
    
    def _renumber_and_reorder_residues(self, atoms, chain_order, positioned_ligs, non_chain_ligs, cap_residue_num):
        """
        Renumber and reorder residues:
        1. Chain residues in order, with positioned LIGs inserted after their coordination residues
        2. Remaining non-positioned LIGs
        3. Cap residue (if present)
        """
        log_message("Renumbering and reordering residues...")
        
        # Build final residue order
        final_order = []
        
        # Insert chain residues and positioned LIGs
        for chain_res_num in chain_order:
            # Skip if this is the cap (will be added at end)
            if cap_residue_num and chain_res_num == cap_residue_num:
                continue
            
            final_order.append(chain_res_num)
            
            # Check if any LIGs should be inserted after this residue
            for lig_num, (after_res, dist) in positioned_ligs.items():
                if after_res == chain_res_num:
                    final_order.append(lig_num)
                    log_message(f"  Inserting LIG {lig_num} after residue {chain_res_num} (coordination)")
        
        # Add remaining non-positioned LIGs
        for lig_num in sorted(non_chain_ligs):
            if lig_num not in positioned_ligs:
                final_order.append(lig_num)
                log_message(f"  Adding non-positioned LIG {lig_num} at end")
        
        # Add cap at the very end
        if cap_residue_num:
            final_order.append(cap_residue_num)
            log_message(f"  Adding cap residue {cap_residue_num} at end")
        
        # Create residue number mapping
        res_num_mapping = {}
        for new_num, old_num in enumerate(final_order, start=1):
            res_num_mapping[old_num] = new_num
            log_message(f"  Residue {old_num} -> {new_num}")
        
        # Apply renumbering to all atoms
        for atom in atoms:
            old_res_num = atom.get('res_num', 1)
            if old_res_num in res_num_mapping:
                atom['res_num'] = res_num_mapping[old_res_num]
        
        # Sort atoms by new residue number, then by original serial
        atoms.sort(key=lambda a: (a['res_num'], a['serial']))
        
        log_message("Residue renumbering and reordering complete")
    
    def _format_pdb_atom_line(self, atom):
        """Format atom dictionary as PDB ATOM line"""
        atom_name = atom.get('atom_name', atom['element'])
        
        # Ensure atom name is max 4 characters
        if len(atom_name) > 4:
            atom_name = atom_name[:4]
        
        # Format with proper spacing for element-based alignment
        if len(atom_name) <= 3:
            formatted_name = f" {atom_name:<3s}"
        else:
            formatted_name = atom_name
        
        line = (f"ATOM  {atom['serial']:>5d} {formatted_name:<4s} "
                f"{atom['res_name']:>3s}  {atom.get('res_num', 1):>4d}    "
                f"{atom['x']:>8.3f}{atom['y']:>8.3f}{atom['z']:>8.3f}"
                f"  1.00  0.00          {atom['element']:>2s}")
        
        return line

    def _check_for_hydrogens(self, pdb_file):
        """Check if the PDB file contains any hydrogen atoms"""
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    element = line[76:78].strip()
                    atom_name = line[12:16].strip()
                    
                    # Check element column first, then atom name
                    if element == 'H' or (not element and atom_name.startswith('H')):
                        return True
        
        return False

    def _add_hydrogens(self, pdb_file, output_dir, case_name, wsl_distro, easyparm_dir):
        """Add hydrogens to the structure using reduce from AmberTools for standard residues,
        then manually add hydrogens to any remaining atoms that need them"""
        log_message("Adding hydrogens with reduce...")
        
        # Create output filename
        protonated_pdb = os.path.join(output_dir, f"{case_name}_protonated.pdb")
        
        # Convert paths to WSL
        wsl_input_pdb = self._convert_to_wsl_path(pdb_file)
        wsl_output_pdb = self._convert_to_wsl_path(protonated_pdb)
        
        # Ensure easyparm_dir doesn't have trailing slash
        easyparm_dir = easyparm_dir.rstrip('/')
        
        # Try to run reduce through pixi environment (same as easyPARM)
        command = (
            f"cd {easyparm_dir} && "
            f"pixi run -e default reduce -BUILD {wsl_input_pdb} > {wsl_output_pdb}"
        )
        
        log_message(f"Executing reduce command: {command}")
        
        reduce_result = subprocess.run(
            ['wsl', '-d', wsl_distro, '--', 'bash', '-ic', command],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # If pixi doesn't work, try sourcing amber.sh
        if reduce_result.returncode != 0:
            log_message("reduce not found in pixi environment, trying AmberTools directly...")
            
            # Try common AmberTools installation paths
            amber_paths = [
                "/usr/local/amber/amber.sh",
                "$HOME/amber/amber.sh",
                "/opt/amber/amber.sh",
                f"{easyparm_dir}/amber.sh"
            ]
            
            success = False
            for amber_path in amber_paths:
                command = (
                    f"source {amber_path} 2>/dev/null && "
                    f"reduce -BUILD {wsl_input_pdb} > {wsl_output_pdb}"
                )
                
                log_message(f"Trying AmberTools source: {amber_path}")
                
                reduce_result = subprocess.run(
                    ['wsl', '-d', wsl_distro, '--', 'bash', '-c', command],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if reduce_result.returncode == 0:
                    success = True
                    log_message(f"Successfully activated AmberTools from: {amber_path}")
                    break
            
            if not success:
                raise NodeException(
                    "protonation",
                    f"Could not find or activate reduce from AmberTools. "
                    f"Please ensure AmberTools is installed in WSL and either:\n"
                    f"1. Available in the pixi environment at {easyparm_dir}\n"
                    f"2. Installed with amber.sh in a standard location\n"
                    f"Last error: {reduce_result.stderr}"
                )
        
        # Log output (reduce often sends info to stderr even on success)
        if reduce_result.stdout:
            log_message(f"reduce stdout:\n{reduce_result.stdout}")
        if reduce_result.stderr:
            log_message(f"reduce stderr:\n{reduce_result.stderr}")
        
        # Verify the protonated file was created
        if not os.path.exists(protonated_pdb):
            raise NodeException(
                "protonation",
                f"Protonated PDB file was not created: {protonated_pdb}"
            )
        
        log_message(f"Successfully added hydrogens with reduce. Output: {protonated_pdb}")
        
        # Now manually add hydrogens to atoms that still need them
        log_message("Checking for atoms that still need hydrogens...")
        final_pdb = self._add_missing_hydrogens_manually(protonated_pdb, output_dir, case_name)
        
        return final_pdb

    def _add_missing_hydrogens_manually(self, pdb_file, output_dir, case_name):
        """Manually add hydrogens to heavy atoms that don't have enough hydrogens"""
        import numpy as np
        
        log_message("Analyzing structure for missing hydrogens...")
        
        # Read PDB and build atom connectivity
        atoms = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    atom_data = {
                        'line': line,
                        'serial': int(line[6:11].strip()),
                        'name': line[12:16].strip(),
                        'resName': line[17:20].strip(),
                        'resSeq': int(line[22:26].strip()),
                        'x': float(line[30:38].strip()),
                        'y': float(line[38:46].strip()),
                        'z': float(line[46:54].strip()),
                        'element': line[76:78].strip() if len(line) > 77 else line[12:16].strip()[0],
                    }
                    atoms.append(atom_data)
        
        # Build connectivity based on distance
        bonds = []
        bond_dict = {i: [] for i in range(len(atoms))}  # Store bonded atom indices
        
        for i, atom1 in enumerate(atoms):
            if atom1['element'] == 'H':
                continue
            for j, atom2 in enumerate(atoms[i+1:], start=i+1):
                if atom2['element'] == 'H':
                    continue
                
                dx = atom1['x'] - atom2['x']
                dy = atom1['y'] - atom2['y']
                dz = atom1['z'] - atom2['z']
                dist = np.sqrt(dx**2 + dy**2 + dz**2)
                
                # Element-specific bond length cutoffs for better accuracy
                elem1 = atom1['element']
                elem2 = atom2['element']
                
                # Typical bond lengths: C-C ~1.4-1.5, C=C ~1.34, C≡C ~1.2
                # Use slightly generous cutoff but not too much
                max_dist = 1.65  # Default
                
                if (elem1 in ['C', 'N', 'O'] and elem2 in ['C', 'N', 'O']):
                    max_dist = 1.6  # C-C, C-N, C-O, N-N, etc.
                elif ('C' in [elem1, elem2] and 'S' in [elem1, elem2]):
                    max_dist = 1.9  # C-S
                
                if dist < max_dist:
                    bonds.append((i, j))
                    bond_dict[i].append(j)
                    bond_dict[j].append(i)
        
        # Detect rings using simple cycle detection
        def is_in_ring(atom_idx, max_ring_size=7):
            """Check if an atom is in a ring using depth-first search"""
            visited = set()
            
            def dfs(current, target, depth, path):
                if depth > max_ring_size:
                    return False
                if depth > 2 and current == target:
                    return True
                if current in visited and current != target:
                    return False
                
                visited.add(current)
                
                for neighbor in bond_dict[current]:
                    if depth > 0 and neighbor == path[-1]:  # Don't go back
                        continue
                    if dfs(neighbor, target, depth + 1, path + [current]):
                        return True
                
                visited.discard(current)
                return False
            
            return dfs(atom_idx, atom_idx, 0, [])
        
        # Detect aromatic/ring atoms
        in_ring = {}
        for i in range(len(atoms)):
            if atoms[i]['element'] in ['C', 'N', 'O']:
                in_ring[i] = is_in_ring(i)
            else:
                in_ring[i] = False
        
        # Count hydrogens bonded to each heavy atom
        h_count = {i: 0 for i in range(len(atoms))}
        for i, atom in enumerate(atoms):
            if atom['element'] == 'H':
                # Find which heavy atom this H is bonded to
                for j, atom2 in enumerate(atoms):
                    if atom2['element'] == 'H':
                        continue
                    dx = atom['x'] - atom2['x']
                    dy = atom['y'] - atom2['y']
                    dz = atom['z'] - atom2['z']
                    dist = np.sqrt(dx**2 + dy**2 + dz**2)
                    if dist < 1.3:  # H bond length cutoff
                        h_count[j] += 1
                        break
        
        # Determine which atoms need more hydrogens
        new_hydrogens = []
        next_serial = max(atom['serial'] for atom in atoms) + 1
        
        for i, atom in enumerate(atoms):
            element = atom['element']
            if element == 'H' or element not in ['C', 'N', 'O', 'S']:
                continue
            
            # Count heavy atom bonds
            heavy_bonds = len(bond_dict[i])
            current_h = h_count[i]
            
            # Calculate expected hydrogens based on element, bonds, and ring membership
            needed_h = 0
            
            if element == 'C':
                if in_ring[i]:
                    # Aromatic/ring carbon
                    if heavy_bonds == 3:  # sp2 carbon in ring (aromatic)
                        needed_h = max(0, 1 - current_h)
                    elif heavy_bonds == 2:  # Should have 2 H (unusual in aromatic rings)
                        needed_h = max(0, 2 - current_h)
                    elif heavy_bonds == 4:  # Saturated ring carbon
                        needed_h = 0  # Already fully bonded
                else:
                    # Non-ring carbon
                    needed_h = max(0, 4 - heavy_bonds - current_h)
            
            elif element == 'N':
                if in_ring[i]:
                    # Nitrogen in ring (like pyridine, pyrrole)
                    if heavy_bonds == 3:  # Quaternary N or protonated
                        needed_h = 0
                    elif heavy_bonds == 2:  # sp2 N in aromatic ring (pyridine-like)
                        needed_h = 0  # Lone pair, no H needed
                else:
                    # Non-ring nitrogen
                    needed_h = max(0, 3 - heavy_bonds - current_h)
            
            elif element == 'O':
                if in_ring[i]:
                    # Oxygen in ring (like furan)
                    needed_h = 0  # Usually doesn't have H
                else:
                    # Non-ring oxygen
                    needed_h = max(0, 2 - heavy_bonds - current_h)
            
            elif element == 'S':
                # Sulfur typically has 2 bonds or lone pairs
                if heavy_bonds >= 2:
                    needed_h = 0
                else:
                    needed_h = max(0, 2 - heavy_bonds - current_h)
            
            if needed_h > 0:
                log_message(f"Adding {needed_h} hydrogen(s) to {element} atom {atom['serial']} "
                        f"(ring={in_ring[i]}, heavy_bonds={heavy_bonds}, current_h={current_h}) "
                        f"in residue {atom['resName']} {atom['resSeq']}")
                
                # Calculate positions for new hydrogens
                # Get vector pointing away from bonded atoms
                bond_vector = np.array([0.0, 0.0, 0.0])
                
                for j in bond_dict[i]:
                    other = atoms[j]
                    vec = np.array([
                        atom['x'] - other['x'],
                        atom['y'] - other['y'],
                        atom['z'] - other['z']
                    ])
                    vec_norm = np.linalg.norm(vec)
                    if vec_norm > 0:
                        bond_vector += vec / vec_norm
                
                # If no bonds found, use a random direction
                if len(bond_dict[i]) == 0:
                    bond_vector = np.array([1.0, 0.0, 0.0])
                else:
                    if np.linalg.norm(bond_vector) < 0.01:
                        bond_vector = np.array([1.0, 0.0, 0.0])
                    else:
                        bond_vector = bond_vector / np.linalg.norm(bond_vector)
                
                # Standard H bond lengths
                bond_length = {'C': 1.09, 'N': 1.01, 'O': 0.96, 'S': 1.34}.get(element, 1.0)
                
                # Add hydrogens
                for h_num in range(needed_h):
                    # Create perpendicular vectors for multiple H atoms
                    angle = (2 * np.pi * h_num) / needed_h
                    
                    # Create perpendicular vectors
                    perp1 = np.array([-bond_vector[1], bond_vector[0], 0])
                    if np.linalg.norm(perp1) < 0.01:
                        perp1 = np.array([0, -bond_vector[2], bond_vector[1]])
                    perp1 = perp1 / np.linalg.norm(perp1)
                    
                    perp2 = np.cross(bond_vector, perp1)
                    perp2 = perp2 / np.linalg.norm(perp2)
                    
                    # Position H
                    if needed_h == 1:
                        h_vec = bond_vector
                    else:
                        h_vec = (bond_vector + 
                                0.5 * (np.cos(angle) * perp1 + np.sin(angle) * perp2))
                        h_vec = h_vec / np.linalg.norm(h_vec)
                    
                    h_pos = np.array([atom['x'], atom['y'], atom['z']]) + bond_length * h_vec
                    
                    new_hydrogens.append({
                        'serial': next_serial,
                        'name': f'H{h_num+1}' if needed_h > 1 else 'H',
                        'resName': atom['resName'],
                        'resSeq': atom['resSeq'],
                        'x': h_pos[0],
                        'y': h_pos[1],
                        'z': h_pos[2],
                        'element': 'H',
                    })
                    next_serial += 1
        
        if new_hydrogens:
            log_message(f"Adding {len(new_hydrogens)} missing hydrogen(s)")
            
            # Write new PDB with added hydrogens
            output_pdb = os.path.join(output_dir, f"{case_name}_fully_protonated.pdb")
            
            with open(pdb_file, 'r') as fin, open(output_pdb, 'w') as fout:
                for line in fin:
                    if line.startswith(('ATOM', 'HETATM')):
                        fout.write(line)
                    elif line.startswith('END'):
                        # Write new hydrogens before END
                        for h in new_hydrogens:
                            fout.write(
                                f"HETATM{h['serial']:5d}  {h['name']:<3s} {h['resName']:>3s}  "
                                f"{h['resSeq']:4d}    {h['x']:8.3f}{h['y']:8.3f}{h['z']:8.3f}"
                                f"  1.00  0.00          {h['element']:>2s}\n"
                            )
                        fout.write(line)
                    else:
                        fout.write(line)
            
            log_message(f"Fully protonated structure written to: {output_pdb}")
            return output_pdb
        else:
            log_message("No missing hydrogens detected. Structure is complete.")
            return pdb_file

    def _find_lig_residues(self, pdb_file):
        """Find all LIG residue numbers in the PDB file"""
        lig_residues = set()
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    res_name = line[17:20].strip()
                    if res_name == 'LIG':
                        res_num = int(line[22:26].strip())
                        lig_residues.add(res_num)
        
        return sorted(list(lig_residues))

    def _create_single_lig_pdb(self, input_pdb, target_lig_resnum, all_lig_residues, output_dir, case_name):
        """
        Create a PDB with only one LIG residue, converting others to ALA.
        Keeps hydrogens bound to ALA heavy atoms and converts CG to HB3.
        """
        log_message(f"Creating PDB with only LIG {target_lig_resnum}")
        
        output_pdb = os.path.join(output_dir, f"{case_name}_LIG_{target_lig_resnum}_temp.pdb")
        
        # Standard ALA atoms to keep (including hydrogens)
        ala_backbone = {'N', 'CA', 'C', 'O', 'CB'}
        ala_hydrogens = {'H', 'HA', 'HB1', 'HB2', 'HB3', 'H1', 'H2', 'H3'}  # Added H1, H2, H3 for N-terminus
        ala_atoms = ala_backbone | ala_hydrogens
        
        output_lines = []
        
        with open(input_pdb, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    res_name = line[17:20].strip()
                    res_num = int(line[22:26].strip())
                    atom_name = line[12:16].strip()
                    
                    # Check if this is a LIG residue to convert
                    if res_name == 'LIG' and res_num in all_lig_residues and res_num != target_lig_resnum:
                        # Convert LIG to ALA
                        if atom_name in ala_atoms:
                            # Keep this atom, change residue name to ALA
                            modified_line = line[:17] + 'ALA' + line[20:]
                            output_lines.append(modified_line)
                        elif atom_name == 'CG':
                            # Convert CG to HB3
                            modified_line = line[:12] + ' HB3' + line[16:17] + 'ALA' + line[20:]
                            output_lines.append(modified_line)
                        # Skip all other atoms
                    else:
                        # Keep all other lines as-is
                        output_lines.append(line)
                elif line.startswith(('TER', 'END')):
                    output_lines.append(line)
        
        # Renumber atoms
        renumbered_lines = self._renumber_atoms(output_lines)
        
        with open(output_pdb, 'w') as f:
            f.writelines(renumbered_lines)
        
        log_message(f"Created temporary PDB: {output_pdb}")
        return output_pdb

    def _renumber_atoms(self, lines):
        """Renumber all atoms sequentially starting from 1"""
        renumbered_lines = []
        atom_number = 1
        
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                renumbered_line = line[:6] + f"{atom_number:5d}" + line[11:]
                renumbered_lines.append(renumbered_line)
                atom_number += 1
            else:
                renumbered_lines.append(line)
        
        return renumbered_lines

    def _run_easyparm_no_lig(self, pdb_file, case_name, output_dir, easyparm_dir, wsl_distro):
        """
        Run easyPARM for a structure without LIG residues and return the path to the generated XYZ file
        """
        log_message("Running easyPARM for structure without LIG residues")
        
        # Get PDB filename
        pdb_filename = os.path.basename(pdb_file)
        
        # Convert paths to WSL
        wsl_pdb_path = self._convert_to_wsl_path(pdb_file)
        
        # Ensure easyparm_dir doesn't have trailing slash
        easyparm_dir = easyparm_dir.rstrip('/')
        
        log_message(f"WSL PDB path: {wsl_pdb_path}")
        log_message(f"WSL easyPARM directory: {easyparm_dir}")
        
        # Build command: copy PDB to easyPARM dir, run easyPARM with input "2" and filename
        command = (
            f"cp {wsl_pdb_path} {easyparm_dir}/ && "
            f"cd {easyparm_dir} && "
            f"echo -e '2\\n{pdb_filename}' | pixi run -e default easyPARM"
        )
        
        log_message(f"Executing command: {command}")
        
        # Execute in WSL
        easyparm_result = subprocess.run(
            ['wsl', '-d', wsl_distro, '--', 'bash', '-ic', command],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Log output
        if easyparm_result.stdout:
            log_message(f"easyPARM stdout:\n{easyparm_result.stdout}")
        if easyparm_result.stderr:
            log_message(f"easyPARM stderr:\n{easyparm_result.stderr}")
        
        if easyparm_result.returncode != 0:
            raise NodeException(
                "execution",
                f"easyPARM execution failed with return code {easyparm_result.returncode}:\n{easyparm_result.stderr}"
            )
        
        log_message("easyPARM completed successfully")
        
        # Copy the generated initial_structure.xyz to output directory with proper naming
        output_xyz_name = f"{case_name}_initial_structure.xyz"
        output_xyz_path = os.path.join(output_dir, output_xyz_name)
        
        # Copy file from WSL to Windows output directory
        wsl_xyz_source = f"{easyparm_dir}/initial_structure.xyz"
        wsl_xyz_dest = self._convert_to_wsl_path(output_xyz_path)
        
        copy_command = f"cp {wsl_xyz_source} {wsl_xyz_dest}"
        
        log_message(f"Copying XYZ file: {copy_command}")
        
        copy_result = subprocess.run(
            ['wsl', '-d', wsl_distro, '--', 'bash', '-c', copy_command],
            capture_output=True,
            text=True
        )
        
        if copy_result.returncode != 0:
            raise NodeException(
                "execution",
                f"Failed to copy XYZ file: {copy_result.stderr}"
            )
        
        # Verify the file was copied
        if not os.path.exists(output_xyz_path):
            raise NodeException(
                "execution",
                f"Expected output XYZ file not found: {output_xyz_path}"
            )
        
        log_message(f"XYZ file copied to: {output_xyz_path}")
        
        # Clean up easyPARM directory - remove all generated files
        cleanup_command = f"cd {easyparm_dir} && rm -f initial_structure.xyz {pdb_filename} *.log *.out *.mol2 *.frcmod *.lib"
        
        log_message("Cleaning up easyPARM directory")
        
        cleanup_result = subprocess.run(
            ['wsl', '-d', wsl_distro, '--', 'bash', '-c', cleanup_command],
            capture_output=True,
            text=True
        )
        
        if cleanup_result.returncode != 0:
            log_message(f"Warning: Cleanup had issues: {cleanup_result.stderr}")
        else:
            log_message("easyPARM directory cleaned successfully")
        
        return output_xyz_path

    def _run_easyparm(self, pdb_file, lig_resnum, case_name, output_dir, easyparm_dir, wsl_distro):
        """
        Run easyPARM for a single PDB file and return the path to the generated XYZ file
        """
        log_message(f"Running easyPARM for LIG {lig_resnum}")
        
        # Get PDB filename
        pdb_filename = os.path.basename(pdb_file)
        
        # Convert paths to WSL
        wsl_pdb_path = self._convert_to_wsl_path(pdb_file)
        wsl_output_dir = self._convert_to_wsl_path(output_dir)
        
        # Ensure easyparm_dir doesn't have trailing slash
        easyparm_dir = easyparm_dir.rstrip('/')
        
        log_message(f"WSL PDB path: {wsl_pdb_path}")
        log_message(f"WSL easyPARM directory: {easyparm_dir}")
        
        # Build command: copy PDB to easyPARM dir, run easyPARM with input "2" and filename
        command = (
            f"cp {wsl_pdb_path} {easyparm_dir}/ && "
            f"cd {easyparm_dir} && "
            f"echo -e '2\\n{pdb_filename}' | pixi run -e default easyPARM"
        )
        
        log_message(f"Executing command: {command}")
        
        # Execute in WSL
        easyparm_result = subprocess.run(
            ['wsl', '-d', wsl_distro, '--', 'bash', '-ic', command],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Log output
        if easyparm_result.stdout:
            log_message(f"easyPARM stdout:\n{easyparm_result.stdout}")
        if easyparm_result.stderr:
            log_message(f"easyPARM stderr:\n{easyparm_result.stderr}")
        
        if easyparm_result.returncode != 0:
            raise NodeException(
                "execution",
                f"easyPARM execution failed with return code {easyparm_result.returncode}:\n{easyparm_result.stderr}"
            )
        
        log_message(f"easyPARM completed successfully for LIG {lig_resnum}")
        
        # Copy the generated initial_structure.xyz to output directory with proper naming
        output_xyz_name = f"{case_name}_LIG_{lig_resnum}_initial_structure.xyz"
        output_xyz_path = os.path.join(output_dir, output_xyz_name)
        
        # Copy file from WSL to Windows output directory
        wsl_xyz_source = f"{easyparm_dir}/initial_structure.xyz"
        wsl_xyz_dest = self._convert_to_wsl_path(output_xyz_path)
        
        copy_command = f"cp {wsl_xyz_source} {wsl_xyz_dest}"
        
        log_message(f"Copying XYZ file: {copy_command}")
        
        copy_result = subprocess.run(
            ['wsl', '-d', wsl_distro, '--', 'bash', '-c', copy_command],
            capture_output=True,
            text=True
        )
        
        if copy_result.returncode != 0:
            raise NodeException(
                "execution",
                f"Failed to copy XYZ file: {copy_result.stderr}"
            )
        
        # Verify the file was copied
        if not os.path.exists(output_xyz_path):
            raise NodeException(
                "execution",
                f"Expected output XYZ file not found: {output_xyz_path}"
            )
        
        log_message(f"XYZ file copied to: {output_xyz_path}")
        
        # Clean up easyPARM directory - remove all generated files
        cleanup_command = f"cd {easyparm_dir} && rm -f initial_structure.xyz {pdb_filename} *.log *.out *.mol2 *.frcmod *.lib"
        
        log_message(f"Cleaning up easyPARM directory")
        
        cleanup_result = subprocess.run(
            ['wsl', '-d', wsl_distro, '--', 'bash', '-c', cleanup_command],
            capture_output=True,
            text=True
        )
        
        if cleanup_result.returncode != 0:
            log_message(f"Warning: Cleanup had issues: {cleanup_result.stderr}")
        else:
            log_message("easyPARM directory cleaned successfully")
        
        return output_xyz_path

    def _convert_to_wsl_path(self, windows_path):
        """Convert Windows path to WSL path format"""
        # Convert C:\Users\... to /mnt/c/Users/...
        if ':' in windows_path:
            drive = windows_path[0].lower()
            path = windows_path[2:].replace('\\', '/')
            return f"/mnt/{drive}{path}"
        return windows_path
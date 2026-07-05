import json
import os
import subprocess
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import AllChem

from bocoflow_core.logger import log_message
from bocoflow_core.node import IONode, NodeException, NodeResult
from bocoflow_core.parameters import (
    BooleanParameter,
    FolderParameter,
    StringParameter,
)


class SMILESToPDBNode(IONode):
    """
    Converts SMILES notation to PDB structure file with bond order information.

    This node uses RDKit to convert a SMILES string into a 3D molecular structure
    in PDB format. The resulting PDB includes CONECT records that preserve bond
    order information (single, double, triple, aromatic bonds). Hydrogens are kept
    implicit (these are often added by other programs later).

    Input: SMILES string
    Output: PDB structure file with bond information
    """

    name = "SMILES to PDB Converter"
    node_key = "SMILESToPDB"
    num_in = 0
    num_out = 1
    color = "#9C27B0"

    OPTIONS = {
        "case_name": StringParameter(
            "Case Name",
            docstring="Name for this molecule (used in output filename)"
        ),
        "smiles": StringParameter(
            "SMILES String",
            docstring="SMILES notation of the molecule (e.g., 'CCO' for ethanol, 'c1ccccc1' for benzene)"
        ),
        "output_dir": FolderParameter(
            "Output Directory",
            docstring="Directory where the PDB file will be saved"
        ),
        "random_seed": StringParameter(
            "Random Seed",
            default="42",
            docstring="Random seed for 3D coordinate generation (for reproducibility)"
        ),
        "force_to_run": BooleanParameter(
            "Force to Run",
            default=False,
            docstring="If true, the node will be executed regardless of the database record"
        ),
    }

    def execute(self, predecessor_data, flow_vars):
        """Execute the SMILES to PDB conversion"""
        log_message(
            f"Starting execution of SMILESToPDB for case: {flow_vars['case_name'].get_value()}"
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
            smiles = flow_vars["smiles"].get_value().strip()
            output_dir = self.resolve_path(flow_vars["output_dir"].get_value())
            random_seed = int(flow_vars["random_seed"].get_value())

            log_message(f"Resolved output directory: {output_dir}")
            result.metadata.update({"output_dir": self.format_output_path(output_dir)})

            # Validate SMILES string
            if not smiles:
                raise NodeException(
                    "execution",
                    "SMILES string cannot be empty"
                )

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Define output path
            output_pdb_path = os.path.join(output_dir, f"{case_name}.pdb")

            log_message(f"Converting SMILES '{smiles}' to PDB structure")

            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                raise NodeException(
                    "execution",
                    f"Invalid SMILES string: {smiles}. Please check the SMILES notation."
                )

            # Add hydrogens temporarily for 3D coordinate generation
            mol_with_h = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            log_message("Generating 3D coordinates...")
            if AllChem.EmbedMolecule(mol_with_h, randomSeed=random_seed) == -1:
                raise NodeException(
                    "execution",
                    "Could not generate 3D coordinates for this molecule. The structure may be too complex or invalid."
                )
            
            # Optimize geometry
            log_message("Optimizing molecular geometry...")
            AllChem.UFFGetMoleculeForceField(mol_with_h).Minimize()
            
            # Remove hydrogens but keep coordinates for heavy atoms
            mol_no_h = Chem.RemoveHs(mol_with_h)
            
            # Write PDB file with bond information
            log_message(f"Writing PDB file to: {output_pdb_path}")
            self._write_pdb_with_bonds(mol_no_h, output_pdb_path)

            # Get molecular properties
            num_atoms = mol_no_h.GetNumAtoms()
            num_bonds = mol_no_h.GetNumBonds()
            molecular_formula = Chem.rdMolDescriptors.CalcMolFormula(mol_no_h)

            log_message(f"Structure created: {num_atoms} atoms, {num_bonds} bonds")
            log_message(f"Molecular formula: {molecular_formula}")

            # Store conversion details
            result.data = {
                "case_name": case_name,
                "smiles": smiles,
                "molecular_formula": molecular_formula,
                "num_heavy_atoms": num_atoms,
                "num_bonds": num_bonds,
                "output_files": {
                    "structure": self.format_output_path(output_pdb_path),
                },
                "working_path": self.format_output_path(output_dir),
            }

            # Set output files
            result.files["output"] = {
                "structure": self.format_output_path(output_pdb_path),
            }

            # Prepare result
            result.success = True
            result.message = f"Successfully converted SMILES to PDB: {case_name} ({molecular_formula}, {num_atoms} atoms)"

            return result.to_json()

        except Exception as e:
            log_message(f"Error in SMILESToPDB: {str(e)}")
            raise NodeException("smiles conversion", str(e))

    def _write_pdb_with_bonds(self, mol, output_file):
        """
        Write molecule to PDB file with CONECT records for bond orders.
        """
        conf = mol.GetConformer()
        
        with open(output_file, 'w') as f:
            # Write ATOM records
            for i, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                atom_name = atom.GetSymbol()
                f.write(f"HETATM{i+1:5d}  {atom_name:<3s} LIG A   1    "
                       f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00          {atom_name:>2s}\n")
            
            # Write CONECT records with bond order information
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx() + 1
                end_idx = bond.GetEndAtomIdx() + 1
                bond_type = bond.GetBondType()
                
                # PDB CONECT format: repeat connections for multiple bonds
                if bond_type == Chem.BondType.SINGLE:
                    f.write(f"CONECT{begin_idx:5d}{end_idx:5d}\n")
                elif bond_type == Chem.BondType.DOUBLE:
                    f.write(f"CONECT{begin_idx:5d}{end_idx:5d}{end_idx:5d}\n")
                elif bond_type == Chem.BondType.TRIPLE:
                    f.write(f"CONECT{begin_idx:5d}{end_idx:5d}{end_idx:5d}{end_idx:5d}\n")
                elif bond_type == Chem.BondType.AROMATIC:
                    # Aromatic bonds represented as 1.5 order (single + double)
                    f.write(f"CONECT{begin_idx:5d}{end_idx:5d}{end_idx:5d}\n")
            
            f.write("END\n")
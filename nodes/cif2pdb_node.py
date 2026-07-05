import os
from datetime import datetime

from bocoflow_core.logger import log_message
from bocoflow_core.node import IONode, NodeException, NodeResult
from bocoflow_core.parameters import (
    BooleanParameter,
    FileParameterEdit,
    FolderParameter,
    StringParameter,
)


class CIF2PDB(IONode):
    """
    Converts mmCIF files to PDB format using BioPython.

    This node uses the BioPython library to convert crystallographic CIF files
    to the legacy PDB format.

    Input: CIF file (typically from structure prediction like Boltz2)
    Output: PDB file with complete structural information

    REQUIREMENTS:
    This node requires BioPython to be installed in your Python environment.
    
    Installation:
        pip install biopython
    
    Or with conda:
        conda install -c conda-forge biopython
    """

    name = "CIF to PDB Converter"
    node_key = "CIF2PDB"
    num_in = 1
    num_out = 1
    color = "#9B59B6"

    OPTIONS = {
        "case_name": StringParameter(
            "Case Name",
            docstring="Name of the case/system for CIF conversion"
        ),
        "input_cif": FileParameterEdit(
            "Input CIF File",
            docstring="CIF file to convert (e.g., from Boltz2 prediction)"
        ),
        "output_dir": FolderParameter(
            "Output Directory",
            docstring="Output directory for PDB file"
        ),
        "preserve_numbering": BooleanParameter(
            "Preserve Residue Numbering",
            default=True,
            docstring="Preserve original residue numbering from CIF file"
        ),
        "force_to_run": BooleanParameter(
            "Force to Run",
            default=False,
            docstring="If true, the node will be executed regardless of the database record"
        ),
    }

    def _check_biopython_installed(self):
        """Check if BioPython is installed and provide helpful error message"""
        try:
            import Bio
            from Bio.PDB import MMCIFParser, PDBIO
            return True
        except ImportError as e:
            error_msg = (
                "\n" + "="*70 + "\n"
                "BioPython is not installed!\n\n"
                "This node requires BioPython to convert CIF files to PDB format.\n\n"
                "To install BioPython, run one of these commands:\n\n"
                "  Using pip:\n"
                "    pip install biopython\n\n"
                "  Using conda:\n"
                "    conda install -c conda-forge biopython\n\n"
                "  Using pip with specific version:\n"
                "    pip install biopython>=1.79\n\n"
                "After installation, restart your Python environment and try again.\n"
                "="*70 + "\n"
            )
            raise NodeException("missing_dependency", error_msg)

    def _convert_cif_to_pdb(self, input_cif, output_pdb, preserve_numbering=True):
        """Convert CIF to PDB using BioPython"""
        try:
            from Bio.PDB import MMCIFParser, PDBIO
            
            # Parse the mmCIF file
            log_message("Parsing mmCIF file with BioPython...")
            parser = MMCIFParser(QUIET=True)
            
            # Use filename without extension as structure ID
            structure_id = os.path.splitext(os.path.basename(input_cif))[0]
            structure = parser.get_structure(structure_id, input_cif)
            
            log_message(f"Structure parsed successfully: {structure_id}")
            
            # Write to PDB format
            log_message("Writing PDB file...")
            io = PDBIO()
            io.set_structure(structure)
            
            # Save with or without preserving numbering
            if preserve_numbering:
                io.save(output_pdb, preserve_atom_numbering=True)
            else:
                io.save(output_pdb)
            
            log_message("PDB file written successfully")
            
            return structure
            
        except ImportError:
            # This should be caught earlier, but just in case
            raise NodeException(
                "missing_dependency",
                "BioPython import failed during conversion. Please install BioPython."
            )
        except Exception as e:
            # Handle BioPython-specific parsing errors
            if "PDBConstructionException" in str(type(e).__name__):
                raise NodeException(
                    "execution",
                    f"BioPython could not parse the CIF file structure: {str(e)}\n"
                    "The CIF file may be corrupted or not follow mmCIF specification."
                )
            else:
                raise NodeException(
                    "execution",
                    f"Error during CIF to PDB conversion: {str(e)}"
                )

    def _extract_structure_stats(self, structure):
        """Extract statistics from BioPython structure object"""
        atom_count = 0
        residue_count = 0
        chain_ids = set()
        models = []
        
        for model in structure:
            models.append(model.id)
            for chain in model:
                chain_ids.add(chain.id)
                for residue in chain:
                    residue_count += 1
                    for atom in residue:
                        atom_count += 1
        
        return {
            "total_atoms": atom_count,
            "total_residues": residue_count,
            "chains": sorted(list(chain_ids)),
            "num_chains": len(chain_ids),
            "num_models": len(models),
            "model_ids": models
        }

    def execute(self, predecessor_data, flow_vars):
        """Execute the CIF to PDB conversion"""
        log_message(
            f"Starting execution of CIF2PDB for case: {flow_vars['case_name'].get_value()}"
        )
        
        # Check BioPython installation first
        self._check_biopython_installed()
        
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
            input_cif = self.resolve_path(flow_vars["input_cif"].get_value())
            output_dir = self.resolve_path(flow_vars["output_dir"].get_value())
            preserve_numbering = flow_vars["preserve_numbering"].get_value()

            log_message(f"Resolved input CIF: {input_cif}")
            log_message(f"Resolved output directory: {output_dir}")
            log_message(f"Preserve residue numbering: {preserve_numbering}")

            result.metadata.update(
                {"output_dir": self.format_output_path(output_dir)}
            )

            # Validate input file
            if not os.path.exists(input_cif):
                raise NodeException(
                    "execution", f"Input CIF file not found: {input_cif}"
                )

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Create output file path
            output_pdb = os.path.join(output_dir, f"{case_name}_structure.pdb")

            # Convert CIF to PDB using BioPython
            log_message("Converting CIF to PDB format using BioPython...")
            structure = self._convert_cif_to_pdb(input_cif, output_pdb, preserve_numbering)
            log_message(f"PDB file created: {output_pdb}")

            # Verify output file was created
            if not os.path.exists(output_pdb):
                raise NodeException(
                    "execution",
                    "PDB output file was not created. Check logs for errors."
                )

            # Extract statistics from structure
            stats = self._extract_structure_stats(structure)
            log_message(
                f"Structure statistics: {stats['total_atoms']} atoms, "
                f"{stats['total_residues']} residues, {stats['num_chains']} chain(s), "
                f"{stats['num_models']} model(s)"
            )

            # Get file size
            file_size = os.path.getsize(output_pdb)
            
            # Record input files
            result.files["input"].update(
                {
                    "input_cif": self.format_output_path(input_cif),
                }
            )

            # Store processing results
            result.data = {
                "case_name": case_name,
                "statistics": stats,
                "output_files": {
                    "pdb": self.format_output_path(output_pdb),
                },
                "file_size_bytes": file_size,
                "preserve_numbering": preserve_numbering,
                "working_path": self.format_output_path(output_dir),
                "biopython_version": self._get_biopython_version(),
            }

            # Set output files
            result.files["output"] = {
                "pdb": self.format_output_path(output_pdb),
            }

            result.success = True
            result.message = (
                f"CIF to PDB conversion completed successfully for {case_name}\n"
                f"Statistics: {stats['total_atoms']} atoms, {stats['total_residues']} residues, "
                f"{stats['num_chains']} chain(s), {stats['num_models']} model(s)"
            )

            return result.to_json()

        except NodeException:
            # Re-raise NodeExceptions as-is
            raise
        except Exception as e:
            log_message(f"Error in CIF2PDB: {str(e)}")
            raise NodeException("cif2pdb conversion", str(e))

    def _get_biopython_version(self):
        """Get BioPython version for metadata"""
        try:
            import Bio
            return Bio.__version__
        except:
            return "unknown"

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import subprocess
import tempfile
import os


class RNAStructureFeatures:
    """Extracts RNA secondary structure features using ViennaRNA."""
    
    def __init__(self, temperature: float = 37.0):
        self.temperature = temperature
        self._check_vienna_rna()
    
    def _check_vienna_rna(self) -> None:
        """Check if ViennaRNA is available."""
        try:
            subprocess.run(['RNAfold', '--version'], 
                         capture_output=True, check=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ViennaRNA not found. Please install ViennaRNA package.")
    
    def extract_features(self, sequences: List[str]) -> pd.DataFrame:
        """Extract structure features for a list of RNA sequences."""
        features = []
        
        for seq in sequences:
            try:
                seq_features = self._compute_sequence_features(seq)
                features.append(seq_features)
            except Exception as e:
                print(f"Warning: Failed to compute features for sequence: {e}")
                features.append(self._get_default_features())
        
        return pd.DataFrame(features)
    
    def _compute_sequence_features(self, sequence: str) -> Dict[str, float]:
        """Compute structure features for a single sequence."""
        mfe, structure = self._fold_sequence(sequence)
        
        features = {
            'mfe': mfe,
            'mfe_normalized': mfe / len(sequence),  # MFE per nucleotide
            'hairpin_count': self._count_hairpins(structure),
            'stem_count': self._count_stems(structure),
            'gc_content': self._calculate_gc_content(sequence),
            'sequence_length': len(sequence),
            'bulge_count': self._count_bulges(structure),
            'loop_count': self._count_loops(structure)
        }
        
        return features
    
    def _fold_sequence(self, sequence: str) -> tuple:
        """Fold RNA sequence using RNAfold."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            f.write(f">seq\n{sequence}\n")
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['RNAfold', '--noPS', f'--temp={self.temperature}'],
                input=sequence,
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                structure_line = lines[1]
                parts = structure_line.split()
                structure = parts[0]
                mfe = float(parts[1].strip('()'))
                return mfe, structure
            else:
                raise ValueError("Unexpected RNAfold output format")
                
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _count_hairpins(self, structure: str) -> int:
        """Count hairpin loops in dot-bracket structure."""
        hairpins = 0
        i = 0
        while i < len(structure):
            if structure[i] == '(':
                # Find matching closing bracket
                bracket_count = 1
                j = i + 1
                while j < len(structure) and bracket_count > 0:
                    if structure[j] == '(':
                        bracket_count += 1
                    elif structure[j] == ')':
                        bracket_count -= 1
                    j += 1
                
                # Check if this is a hairpin (consecutive dots between brackets)
                if j > i + 1:
                    inner = structure[i+1:j-1]
                    if '(' not in inner and ')' not in inner and len(inner) >= 3:
                        hairpins += 1
                
                i = j
            else:
                i += 1
        
        return hairpins
    
    def _count_stems(self, structure: str) -> int:
        """Count stem regions in dot-bracket structure."""
        stems = 0
        in_stem = False
        
        for i in range(len(structure) - 1):
            current_paired = structure[i] in '()'
            next_paired = structure[i + 1] in '()'
            
            if current_paired and not in_stem:
                stems += 1
                in_stem = True
            elif not current_paired and in_stem:
                in_stem = False
        
        return stems
    
    def _count_bulges(self, structure: str) -> int:
        """Count bulge loops in structure."""
        bulges = 0
        i = 0
        while i < len(structure):
            if structure[i] == '.':
                # Count consecutive dots
                dot_count = 0
                start = i
                while i < len(structure) and structure[i] == '.':
                    dot_count += 1
                    i += 1
                
                # Check if surrounded by paired bases (bulge)
                if (start > 0 and start + dot_count < len(structure) and
                    structure[start - 1] in '()' and structure[start + dot_count] in '()'):
                    if 1 <= dot_count <= 10:  # Typical bulge size
                        bulges += 1
            else:
                i += 1
        
        return bulges
    
    def _count_loops(self, structure: str) -> int:
        """Count all loop regions."""
        loops = 0
        in_loop = False
        
        for char in structure:
            if char == '.' and not in_loop:
                loops += 1
                in_loop = True
            elif char in '()' and in_loop:
                in_loop = False
        
        return loops
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content of sequence."""
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence) if len(sequence) > 0 else 0.0
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when computation fails."""
        return {
            'mfe': 0.0,
            'mfe_normalized': 0.0,
            'hairpin_count': 0,
            'stem_count': 0,
            'gc_content': 0.0,
            'sequence_length': 0,
            'bulge_count': 0,
            'loop_count': 0
        }
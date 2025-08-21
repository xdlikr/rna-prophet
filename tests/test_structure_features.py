import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.features.structure import RNAStructureFeatures


class TestRNAStructureFeatures:
    
    def setup_method(self):
        # Mock ViennaRNA check to avoid dependency in tests
        with patch.object(RNAStructureFeatures, '_check_vienna_rna'):
            self.structure_extractor = RNAStructureFeatures()
    
    def test_gc_content_calculation(self):
        gc_content = self.structure_extractor._calculate_gc_content('ACGU')
        assert gc_content == 0.5  # 2 out of 4 are G or C
        
        gc_content = self.structure_extractor._calculate_gc_content('AAUU')
        assert gc_content == 0.0  # No G or C
    
    def test_hairpin_counting(self):
        # Simple hairpin structure
        structure = '(((...)))'
        hairpins = self.structure_extractor._count_hairpins(structure)
        assert hairpins == 1
        
        # No hairpins
        structure = '((()))'
        hairpins = self.structure_extractor._count_hairpins(structure)
        assert hairpins == 0
    
    def test_stem_counting(self):
        structure = '(((...)))'
        stems = self.structure_extractor._count_stems(structure)
        assert stems >= 1
        
        structure = '......'
        stems = self.structure_extractor._count_stems(structure)
        assert stems == 0
    
    def test_default_features(self):
        default = self.structure_extractor._get_default_features()
        expected_keys = [
            'mfe', 'mfe_normalized', 'hairpin_count', 'stem_count',
            'gc_content', 'sequence_length', 'bulge_count', 'loop_count'
        ]
        assert set(default.keys()) == set(expected_keys)
        assert all(isinstance(v, (int, float)) for v in default.values())
    
    @patch('subprocess.run')
    def test_fold_sequence(self, mock_run):
        # Mock successful RNAfold execution
        mock_run.return_value = MagicMock(
            stdout='ACGU\n((.)) (-5.2)',
            returncode=0
        )
        
        mfe, structure = self.structure_extractor._fold_sequence('ACGU')
        assert mfe == -5.2
        assert structure == '((.))'
        
        mock_run.assert_called_once()
    
    @patch.object(RNAStructureFeatures, '_fold_sequence')
    def test_extract_features(self, mock_fold):
        # Mock folding results
        mock_fold.return_value = (-10.5, '(((...)))')
        
        sequences = ['ACGUACGU', 'UGCAUGCA']
        features_df = self.structure_extractor.extract_features(sequences)
        
        assert len(features_df) == 2
        assert 'mfe' in features_df.columns
        assert 'gc_content' in features_df.columns
        assert 'sequence_length' in features_df.columns
    
    @patch.object(RNAStructureFeatures, '_compute_sequence_features')
    def test_extract_features_with_error(self, mock_compute):
        # Mock error in feature computation
        mock_compute.side_effect = Exception("Folding error")
        
        sequences = ['ACGU']
        features_df = self.structure_extractor.extract_features(sequences)
        
        assert len(features_df) == 1
        # Should use default features when error occurs
        assert features_df.iloc[0]['mfe'] == 0.0
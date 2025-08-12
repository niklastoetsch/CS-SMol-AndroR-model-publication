#!/usr/bin/env python3
"""
Test script to verify that all dependencies are correctly installed.

Run this script after installation to check that everything is working properly.
"""

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        print("✓ RDKit imported successfully")
    except ImportError as e:
        print(f"✗ RDKit import failed: {e}")
        print("  Try: conda install -c conda-forge rdkit")
        return False
    
    try:
        import tqdm
        print("✓ tqdm imported successfully")
    except ImportError as e:
        print(f"✗ tqdm import failed: {e}")
        return False
    
    return True


def test_rdkit_functionality():
    """Test basic RDKit functionality."""
    print("\nTesting RDKit functionality...")
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        # Test SMILES parsing
        mol = Chem.MolFromSmiles("CCO")
        if mol is None:
            print("✗ RDKit SMILES parsing failed")
            return False
        print("✓ RDKit SMILES parsing works")
        
        # Test fingerprint generation
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
        if fp is None:
            print("✗ RDKit fingerprint generation failed")
            return False
        print("✓ RDKit fingerprint generation works")
        
        return True
        
    except Exception as e:
        print(f"✗ RDKit functionality test failed: {e}")
        return False


def test_local_modules():
    """Test that local modules can be imported."""
    print("\nTesting local module imports...")
    
    try:
        import ml
        print("✓ ml module imported successfully")
    except ImportError as e:
        print(f"✗ ml module import failed: {e}")
        return False
    
    try:
        import analysis
        print("✓ analysis module imported successfully")
    except ImportError as e:
        print(f"✗ analysis module import failed: {e}")
        return False
    
    try:
        import utils
        print("✓ utils module imported successfully")
    except ImportError as e:
        print(f"✗ utils module import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality of the pipeline."""
    print("\nTesting basic functionality...")
    
    try:
        from ml import create_pipeline
        pipeline = create_pipeline()
        print("✓ Pipeline creation works")
    except Exception as e:
        print(f"✗ Pipeline creation failed: {e}")
        return False
    
    try:
        from utils import FP_COLUMNS
        assert len(FP_COLUMNS) == 1024
        print("✓ Fingerprint columns defined correctly")
    except Exception as e:
        print(f"✗ Fingerprint columns test failed: {e}")
        return False
    
    try:
        import pandas as pd
        from utils import add_fingerprints_to_df
        
        df = pd.DataFrame({'flat_smiles': ['CCO', 'CC(=O)O']})
        df_with_fps = add_fingerprints_to_df(df)
        assert 'fp_0' in df_with_fps.columns
        print("✓ Fingerprint generation works")
    except Exception as e:
        print(f"✗ Fingerprint generation test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("AndroR Model Installation Test")
    print("=" * 30)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test RDKit functionality
    if not test_rdkit_functionality():
        success = False
    
    # Test local modules
    if not test_local_modules():
        success = False
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    print("\n" + "=" * 30)
    if success:
        print("✓ All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("1. Run 'python example_usage.py' for a usage example")
        print("2. Start Jupyter and run the analysis notebooks")
        print("3. See README.md for more information")
    else:
        print("✗ Some tests failed. Please check the installation.")
        print("\nTroubleshooting:")
        print("1. Ensure all packages are installed: pip install -r requirements.txt")
        print("2. For RDKit issues, try: conda install -c conda-forge rdkit")
        print("3. Check that you're in the correct directory")
        print("4. See INSTALL.md for detailed installation instructions")
    
    return success


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, Descriptors3D, PandasTools, Draw
from rdkit.Chem.rdchem import Atom
from rdkit.Chem.MolStandardize import rdMolStandardize
#import molvs
#from molvs.fragment import LargestFragmentChooser
import logging
import os


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chemistry_standardization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def read_input_data(filepath):
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Log basic information about the DataFrame
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Columns found: {df.columns.tolist()}")
        
        # Show first few rows
        logger.info("\nFirst few rows of the DataFrame:")
        logger.info(f"\n{df.head()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        raise


def standardize_smiles_rdkit(df):
    try:
        # Function to standardize a single SMILES using RDKit standardization
        def standardize_mol_rdkit(smiles):
            # Source: https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/MolStandardize/TransformCatalog/normalizations.in    
            RDLogger.DisableLog('rdApp.info')      
            smiles_std = None
            try:
                mol = Chem.MolFromSmiles(smiles)
                #print(mol.GetNumAtoms())
                if mol is not None:
                    Chem.SanitizeMol(mol,sanitizeOps=(Chem.SANITIZE_ALL^Chem.SANITIZE_CLEANUP^Chem.SANITIZE_PROPERTIES))
                    cm = rdMolStandardize.Normalize(mol)
                
                    uncharger = rdMolStandardize.Uncharger()
                    um = uncharger.uncharge(cm)
                
                    im = rdMolStandardize.Reionize(um)
                
                    lm = rdMolStandardize.FragmentParent(im)

                    #te = rdMolStandardize.TautomerEnumerator()  
                    #std_mol = te.Canonicalize(lm)
                    
                    # get smiles for standardized molecule
                    #smiles_std = Chem.MolToSmiles(std_mol, canonical=True)
                    smiles_std = Chem.MolToSmiles(lm, canonical=True)
                else:
                    smiles_std = 'remove'   
            finally:
                return smiles_std
        
        # Create new column with standardized SMILES
        df['std_smiles'] = df['Structure'].apply(standardize_mol_rdkit)
        
        # Print a few examples to verify
        print("\nFirst few standardized SMILES:")
        print(df[['Structure', 'std_smiles']].head())
        
        return df
    
        # Count successful conversions
        successful = df['std_smiles'].notna().sum()
        failed = df['std_smiles'].isna().sum()
        print(f"\nStandardization results:")
        print(f"Successfully standardized: {successful}")
        print(f"Failed to standardize: {failed}")
        
    except Exception as e:
        logger.error(f"Error during SMILES standardization: {e}")
        raise


def create_flat_smiles_rdkit(df):
    try:
        # Function to create flat SMILES using RDKit
        def flatten_single_smiles_rdkit(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Remove stereochemistry
                    Chem.RemoveStereochemistry(mol)
                    # Convert to canonical SMILES
                    return Chem.MolToSmiles(mol, canonical=True)
                else:
                    return None
            except:
                return None
        
        # Create new column with flat SMILES
        df['flat_smiles'] = df['std_smiles'].apply(flatten_single_smiles_rdkit)
        
        # Print a few examples to verify
        print("\nFirst few flattened SMILES:")
        print(df[['Structure', 'std_smiles', 'flat_smiles']].head())
        
        return df
    except Exception as e:
        print(f"Error in flattening: {str(e)}")
        return df    
        
        # Count successful conversions
        successful = df['flat_smiles'].notna().sum()
        failed = df['flat_smiles'].isna().sum()
        print(f"\nFlattening results:")
        print(f"Successfully flattened: {successful}")
        print(f"Failed to flatten: {failed}")

    except Exception as e:
        logger.error(f"Error during SMILES flattening: {e}")
        raise

def handle_duplicates(df):
    try:
        print("\nHandling duplicate flat SMILES...")
        
        # Initialize DataFrame for conflicting entries
        conflicts_df = pd.DataFrame(columns=df.columns)
        
        # Find all duplicate flat SMILES
        duplicate_mask = df['flat_smiles'].duplicated(keep=False)
        duplicates = df[duplicate_mask].copy()
        
        if len(duplicates) == 0:
            print("No duplicate flat SMILES found.")
            return df, conflicts_df
        
        print(f"Found {len(duplicates)} rows with duplicate flat SMILES")
        
        # Group duplicates by flat SMILES
        groups = duplicates.groupby('flat_smiles')
        
        # Lists to store indices to keep and conflict rows
        indices_to_keep = []
        conflict_indices = []
        
        for flat_smiles, group in groups:
            unique_classes = group['final class'].unique()
            if len(unique_classes) == 1:
                # No conflicts - keep first occurrence
                indices_to_keep.append(group.index[0])
            else:
                # Conflicts found - add all rows to conflicts
                conflict_indices.extend(group.index)
        
        # Create conflicts DataFrame
        conflicts_df = df.loc[conflict_indices].copy()
        
        # Create clean DataFrame
        clean_df = pd.concat([
            df[~duplicate_mask],  # Non-duplicates
            df.loc[indices_to_keep]  # Non-conflicting duplicates (one instance)
        ])
        
        # Print summary
        print(f"\nDuplicate handling summary:")
        print(f"Rows with conflicting classes: {len(conflicts_df)}")
        print(f"Duplicate rows removed (keeping one): {len(duplicates) - len(conflicts_df) - len(indices_to_keep)}")
        print(f"Original DataFrame size: {len(df)}")
        print(f"Clean DataFrame size: {len(clean_df)}")
        
        return clean_df, conflicts_df
        
    except Exception as e:
        logger.error(f"Error handling duplicates: {e}")
        raise

if __name__ == "__main__":
    input_filepath = "data/Supp_file_S2_Result_Table.csv"
    
    # Create output filename by adding _STD before the extension
    base_path = os.path.splitext(input_filepath)[0]  # Remove extension
    output_filepath = f"{base_path}_STD_rdkit.csv"
    conflicts_filepath = f"{base_path}_conflicts_rdkit.csv"
    
    # Read and inspect the data
    df = read_input_data(input_filepath)
    
    # Standardize SMILES
    df = standardize_smiles_rdkit(df)
    
    # Create flat SMILES
    df = create_flat_smiles_rdkit(df)
    
    # Handle duplicates and get conflicts
    clean_df, conflicts_df = handle_duplicates(df)

    # Save the results
    clean_df.to_csv(output_filepath, index=False)
    conflicts_df.to_csv(conflicts_filepath, index=False)
    
    print(f"\nResults saved:")
    print(f"Clean data saved to: {output_filepath}")
    print(f"Conflicts saved to: {conflicts_filepath}")

    

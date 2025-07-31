import numpy as np
import pandas as pd
from rdkit import Chem
import predefined_models  # This includes your predefined_mordred()
import joblib
import os
import json
from predefined_models import predefined_mordred 

selected_columns = [
    'nHBAcc', 'nHBDon', 'nRot', 'nBonds', 'nAromBond', 'nBondsO', 'nBondsS',
    'TopoPSA(NO)', 'TopoPSA', 'LabuteASA', 'bpol', 'nAcid', 'nBase',
    'ECIndex', 'GGI1', 'SLogP', 'SMR', 'BertzCT', 'BalabanJ', 'Zagreb1',
    'nHRing', 'naHRing', 'NsCH3', 'NaaCH', 'NaaaC', 'NssssC',
    'SsCH3', 'SdCH2', 'SssCH2', 'StCH', 'SdsCH', 'SaaCH', 'SsssCH', 'SdssC',
    'SaasC', 'SaaaC', 'SsNH2', 'SssNH', 'StN', 'SdsN', 'SaaN', 'SsssN',
    'SaasN', 'SsOH', 'SdO', 'SssO', 'SaaO', 'SsF', 'SdsssP', 'SsSH', 'SdS',
    'SddssS', 'SsCl', 'SsI'
]

# Get the available descriptor names once
example_mol = Chem.MolFromSmiles("CC")
available_columns = predefined_models.predefined_mordred(example_mol, "all", desc_names=True)
filtered_columns = [col for col in selected_columns if col in available_columns]

def compute_descriptors(smiles: str) -> pd.DataFrame:
    """
    Compute selected descriptors from SMILES using predefined_mordred().
    
    Returns:
        pd.DataFrame: One-row DataFrame with descriptors as columns
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        mol = Chem.AddHs(mol)  # Add hydrogens like during training

        desc_vector = predefined_models.predefined_mordred(mol, "all")
        desc_dict = dict(zip(available_columns, desc_vector))
        selected_values = [desc_dict.get(col, np.nan) for col in filtered_columns]

        df = pd.DataFrame([selected_values], columns=filtered_columns)

        return df

    except Exception as e:
        print(f" Error processing SMILES '{smiles}': {e}")
        return pd.DataFrame(columns=filtered_columns)



# Load LabelEncoder once globally
LE_PATH = "label_encoder_solvent.pkl"

if not os.path.exists(LE_PATH):
    raise FileNotFoundError(f"LabelEncoder file not found at {LE_PATH}")

label_encoder = joblib.load(LE_PATH)

def encode_solvent(solvent: str) -> int:
    """
    Converts solvent name to its label-encoded integer value using saved LabelEncoder.

    Parameters:
        solvent (str): Name of the solvent (e.g., 'water')

    Returns:
        int: Encoded integer label

    Raises:
        ValueError: If solvent is not found in the LabelEncoder classes
    """
    try:
        encoded = label_encoder.transform([solvent])[0]
        # print(int(encoded))
        return int(encoded)
    except ValueError:
        known = list(label_encoder.classes_)
        raise ValueError(
            f"Solvent '{solvent}' not found in label encoder.\n"
            f"Available solvents: {known}"
        )

# res = encode_solvent('1,2-dichloroethane') # -> output was 0
# res = encode_solvent('DMAc') 
# print("Label encoded value is: ",res)


# === Load persistent assets once ===
with open("feature_columns.json", "r") as f:
    feature_order = json.load(f)

model = joblib.load("model_compressed.joblib")

def predict_solubility(smiles: str, solvent: str, temperature: float = 298.15) -> float:
    """
    Predict LogS from SMILES and solvent using trained model.
    
    Parameters:
        smiles (str): Molecule SMILES
        solvent (str): Solvent name (e.g., "Water")
        temperature (float): Temperature in Kelvin. Default is 298.15 K.
        
    Returns:
        float: Predicted LogS value
    """
    # Step 1: Compute descriptors using prebuilt function
    desc_df = compute_descriptors(smiles)
    
    if desc_df.empty:
        raise ValueError("Descriptor computation failed. Check the SMILES string.")

    # Step 2: Add solvent and temperature
    desc_df["Solvent_encoded"] = encode_solvent(solvent)
    desc_df["Temperature_K"] = temperature

    # Step 3: Reorder columns
    try:
        X = desc_df[feature_order]
    except KeyError as e:
        missing = set(feature_order) - set(desc_df.columns)
        raise KeyError(f"Missing features in input: {missing}") from e

    # Step 4: Predict
    logS = model.predict(X)[0]
    return logS


if __name__ == "__main__":

    # For this i got Predicted LogS: -2.0246 , original -2.177078
    # smiles = "ClC1=CC=C(C=O)C=C1"
    # solvent = "water"
    # temperature = 300.0

    # For this i got Predicted LogS: -3.1175 , original -3.123150
    # smiles = "CC1=CC=CC(C=C)=C1" 
    # solvent = "water"
    # temperature = 300.0

    # For this i got Predicted LogS: 0.6375 , original 0.650142
    smiles = "Cc1c(Br)cccc1C(=O)O" 
    solvent = "DMF"
    temperature = 303.15

    try:
        prediction = predict_solubility(smiles, solvent, temperature)
        print(f" Predicted LogS: {prediction:.4f}")
    except Exception as e:
        print(f" Error: {e}")

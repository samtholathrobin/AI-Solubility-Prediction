# this is used for descriptor of solvent and smile both 110 columns one
import numpy as np
import pandas as pd
from rdkit import Chem
import joblib
import json
import os
from predefined_models import predefined_mordred

# -----------------------------
# Descriptor Selection (solute)
# -----------------------------
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

# -----------------------------
# Load feature order (solute + solvent + T)
# -----------------------------
with open("feature_columns_110.json", "r") as f:
    feature_order = json.load(f)

# -----------------------------
# Load trained model
# -----------------------------
model_path = "model_RF_SolventDescriptors_Compressed_gzip4.joblib"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}")
model = joblib.load(model_path)

# -----------------------------
# Descriptor Calculator
# -----------------------------
example_mol = Chem.MolFromSmiles("CC")
available_columns = predefined_mordred(example_mol, "all", desc_names=True)
filtered_columns = [col for col in selected_columns if col in available_columns]

def compute_descriptors(smiles: str) -> pd.DataFrame:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        mol = Chem.AddHs(mol)

        desc_vector = predefined_mordred(mol, "all")
        desc_dict = dict(zip(available_columns, desc_vector))
        selected_values = [desc_dict.get(col, np.nan) for col in filtered_columns]

        return pd.DataFrame([selected_values], columns=filtered_columns)

    except Exception as e:
        print(f"⚠️ Error processing SMILES '{smiles}': {e}")
        return pd.DataFrame(columns=filtered_columns)

# -----------------------------
# Main Prediction Function
# -----------------------------
def predict_solubility(solute_smiles: str, solvent_smiles: str, temperature: float = 298.15) -> float:
    solute_df = compute_descriptors(solute_smiles)
    solvent_df = compute_descriptors(solvent_smiles)

    # if solute_df.empty or solvent_df.empty:
        # raise ValueError("Descriptor computation failed for solute or solvent.")
    if solute_df.empty:
        raise ValueError("Descriptor computation failed for solute.")
    if solvent_df.empty:
        raise ValueError("Descriptor computation failed for solvent.")


    # Rename solvent columns to match training feature names
    solvent_df.columns = [f"Solvent_{col}" for col in solvent_df.columns]


    # Combine
    combined_df = pd.concat([solute_df, solvent_df], axis=1)
    combined_df["Temperature_K"] = temperature

    try:
        X = combined_df[feature_order]
    except KeyError as e:
        missing = set(feature_order) - set(combined_df.columns)
        raise KeyError(f"❌ Missing features: {missing}")

    prediction = model.predict(X)[0]
    return prediction

# -----------------------------
# Example Testing Block
# -----------------------------
# if __name__ == "__main__":
#     test_cases = [
#         {
#             "solute": "ClC1=CC=C(C=O)C=C1",  # Solute SMILES
#             "solvent": "O=C(N(C)C)C",         # DMF
#             "temp": 300.0,
#         },
#         {
#             "solute": "CC1=CC=CC(C=C)=C1",    # Toluene derivative
#             "solvent": "O=C(C)Oc1ccccc1C(=O)O",  # Ethyl acetate
#             "temp": 298.15,
#         },
#         {
#             "solute": "Cc1c(Br)cccc1C(=O)O",  # Br-benzoic acid
#             "solvent": "CN(C)C=O",            # DMF (simplified)
#             "temp": 303.15,
#         }
#     ]

#     for i, case in enumerate(test_cases):
#         try:
#             logS = predict_solubility(case["solute"], case["solvent"], case["temp"])
#             print(f"[{i+1}] Predicted LogS: {logS:.4f} | Solute: {case['solute']} | Solvent: {case['solvent']}")
#         except Exception as e:
#             print(f"[{i+1}] ❌ Failed: {e}")

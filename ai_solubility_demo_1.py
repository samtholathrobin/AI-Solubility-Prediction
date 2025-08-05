import pubchempy as pcp
import pandas as pd
import streamlit as st
from functions import predict_solubility  # New version using descriptors for solute + solvent

#  Helper: Convert solvent name to SMILES using PubChem
def get_smiles_with_get_properties(compound_name):
    try:
        result = pcp.get_properties('SMILES', compound_name, namespace='name')
        if result and 'SMILES' in result[0]:
            return result[0]['SMILES']
        else:
            return None
    except Exception as e:
        print(f"Error for '{compound_name}': {e}")
        return None

#  Load Experimental Dataset
df = pd.read_csv('Final_Merged_Aqsoldbc_Bigsolv2.csv')
# df = pd.read_csv(r'C:\Users\LENOVO\Desktop\Sekuen\Sekuen1\AISolubility\DataSets\Final_Merged_Aqsoldbc_Bigsolv2.csv')
df = df.convert_dtypes()
solvents_list = [None] + df.Solvent.dropna().unique().tolist()

#  Experimental Matching Function
def search_best_match(df, smiles, solvent=None, temperature_k=None):
    filtered = df[df['SMILES'] == smiles]
    if filtered.empty:
        return None
    if solvent:
        filtered = filtered[filtered['Solvent'] == solvent]
        if filtered.empty:
            return None
    if temperature_k is not None:
        idx = (filtered['Temperature_K'] - temperature_k).abs().idxmin()
        return filtered.loc[[idx]]
    return filtered.sort_values(by="LogS(mol/L)", ascending=False)

#  Streamlit UI
st.title("🔬 Sekuen Solubility Prediction System")
st.write("Enter solute (SMILES), solvent, and temperature to retrieve experimental or predicted LogS.")

smiles_input = st.text_input("Enter Solute SMILES:").replace(" ", "")
solvent_input = st.selectbox("Select Solvent (optional):", solvents_list)
temp_str = st.text_input("Enter Temperature in Kelvin (optional):", "")
# temperature_input = float(temp_str) if temp_str.strip() else None
try:
    temperature_input = float(temp_str.strip()) if temp_str.strip() else None
except ValueError:
    st.error("❌ Invalid temperature input. Please enter a number in Kelvin.")
    st.stop()

if st.button("🔍 Search & Predict"):
    if not smiles_input:
        st.error("❌ Please enter a valid SMILES string.")
    else:
        # 🧪 Experimental match
        result = search_best_match(df, smiles_input, solvent_input, temperature_input)

        # Set defaults
        solvent_name = solvent_input if solvent_input else "water"
        temperature = temperature_input if temperature_input is not None else 298.15

        if not solvent_input:
            st.info("ℹ️ No solvent selected — defaulting to **water**.")
        if temperature_input is None:
            st.info("ℹ️ No temperature provided — defaulting to **298.15 K**.")

        # Get solvent SMILES from name
        # solvent_name = solvent_name.strip().lower()
        solvent_smiles = get_smiles_with_get_properties(solvent_name)
        if not solvent_smiles:
            st.error(f"❌ Could not retrieve SMILES for solvent: '{solvent_name}'")
        else:
            try:
                predicted_logS = predict_solubility(
                    solute_smiles=smiles_input,
                    solvent_smiles=solvent_smiles,
                    temperature=temperature
                )

                output_dict = {
                    "Solute SMILES": smiles_input,
                    "Solvent": solvent_name,
                    "Solvent SMILES": solvent_smiles,
                    "Temperature (K)": temperature,
                    "Predicted LogS": round(predicted_logS, 4)
                }

                if result is not None:
                    st.success("✅ Experimental data found.")
                    st.dataframe(result)
                    output_dict["Experimental LogS"] = result["LogS(mol/L)"].values[0]
                else:
                    st.warning("⚠️ No experimental data found in database.")

                st.subheader("📊 Prediction Result:")
                st.dataframe(pd.DataFrame([output_dict]))

            # except Exception as e:
                # st.error(f"❌ Prediction failed: {e}")
            except Exception as e:
                if "Descriptor computation failed" in str(e):
                    st.error("❌ Descriptor computation failed. This may be due to uncommon atoms or a malformed SMILES string.")
                else:
                    st.error(f"❌ Prediction failed: {e}")

import pandas as pd
import streamlit as st 

# df = pd.read_csv('Final_Merged_Aqsoldbc_Bigsolv2.csv')
df = pd.read_csv(r'C:\Users\LENOVO\Desktop\Sekuen\Sekuen1\AISolubility\DataSets\Final_Merged_Aqsoldbc_Bigsolv2.csv')
df = df.convert_dtypes()
solvents_list = [None]+df.Solvent.unique().tolist()

def search_best_match(df, smiles, solvent=None, temperature_k=None):
    """
    Returns a row from the dataframe based on:
    - SMILES only: returns all matches sorted by LogS descending.
    - SMILES + Solvent: returns all matches with that solvent sorted by closest Temperature_K if provided.
    - SMILES + Solvent + Temperature_K: returns best matching row by minimal temperature difference.
    
    Returns:
        - A single row (Series) if SMILES, Solvent, and Temperature_K are all provided.
        - A filtered DataFrame otherwise.
    """
    # Filter by SMILES
    filtered = df[df['SMILES'] == smiles]
    if filtered.empty:
        return None

    # Filter by solvent if provided
    if solvent:
        filtered = filtered[filtered['Solvent'] == solvent]
        if filtered.empty:
            return None

    # Find best temperature match if temperature is provided
    if temperature_k is not None:
        idx = (filtered['Temperature_K'] - temperature_k).abs().idxmin()
        return filtered.loc[idx]

    # Otherwise, return DataFrame sorted by LogS
    return filtered.sort_values(by="LogS(mol/L)", ascending=False)

st.title("Solubility Prediction System")
st.write("This application allows you to search for solubility data based on SMILES, solvent, and temperature.")
smiles_input = st.text_input("Enter SMILES:").replace(" ","")


solvent_input = st.selectbox("Select Solvent:", solvents_list)
temp_str = st.text_input("Enter Temperature (K) or leave blank:", "")
temperature_input = float(temp_str) if temp_str.strip() else None

if st.button("Search"):
    result = search_best_match(df, smiles_input, solvent_input, temperature_input)

    try:
        from functions import predict_solubility  # prediction logic

        # default values for solvent and temperature added, so both experimental data and predicted solubility are shown — even if the user only enters SMILES:
        solvent_for_prediction = solvent_input if solvent_input else "water"
        temperature_for_prediction = temperature_input if temperature_input is not None else 298.15
        predicted_logS = predict_solubility(smiles_input, solvent_for_prediction, temperature_for_prediction)

        # Format both results in one combined display
        output_dict = {
            "SMILES": smiles_input,
            "Solvent": solvent_for_prediction,
            "Temperature_K": temperature_for_prediction,
            "Predicted LogS": predicted_logS
        }

        if result is not None:
            st.success("✅ Experimental data found.")
            st.dataframe(result)
            output_dict["Experimental LogS"] = result["LogS(mol/L)"].values[0] if "LogS(mol/L)" in result else "Not Found"

        else:
            st.warning("⚠️ No experimental data found.")

        # Show combined summary
        st.write("📊 Prediction:")
        st.dataframe(pd.DataFrame([output_dict]))

    except Exception as e:
        st.error(f"Prediction failed: {e}")

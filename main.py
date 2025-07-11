# main.py

import streamlit as st
import pandas as pd
import sys
import os

# Ensure models/ml.py is importable
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from ml import predict_nccn  # Update if your function has a different name

# ---- App Header ----
st.title("NCCN Guideline Sample Analyzer")
st.write("**Evan Moh** · Duke University · AIPI-540 Deep Learning")

st.markdown(
    """
    ---
    **Instructions:**  
    - Select an example below or paste your own NCCN guideline sentence.  
    - Click **Analyze** to see the model output.
    ---
    """
)

# ---- Sample DataFrame of Predictions ----

sample_data = [
    {
        "sentence": "Preclinical Pharmacology, Efficacy and Safety of Varenicline in Smoking Cessation and Clinical Utility in High Risk Patients.",
        "label": "NSCLC",
        "predicted": "NSCLC",
        "correct": True
    },
    {
        "sentence": "Nivolumab and hyaluronidasenvhy has different dosing and administration instructions compared to IV nivolumab MS1",
        "label": "SCLC",
        "predicted": "NSCLC",
        "correct": False
    },
    {
        "sentence": "Impact of introducing stereotactic lung radiotherapy for elderly patients with stage I non small cell lung cancer: a population based time trend analysis.",
        "label": "NSCLC",
        "predicted": "NSCLC",
        "correct": True
    },
    {
        "sentence": "At least 10% of the tumor should show LCNEC morphology to be classified as combined SCLC and LCNEC. Comprehensive molecular profiling can be considered in rare cases particularly for patients with extensive stage/relapsed SCLC who do not smoke tobacco, lightly smoke (<10 cigarettes/day), have remote smoking history, or have diagnostic or therapeutic dilemma, or at time of relapse if not previously done, because this may change management.",
        "label": "SCLC",
        "predicted": "SCLC",
        "correct": True
    },
    {
        "sentence": "J Natl Compr Canc Netw 2018;16:461 466.",
        "label": "NSCLC",
        "predicted": "SCLC",
        "correct": False
    },
    {
        "sentence": "Radiation dosevolume effects in the esophagus.",
        "label": "NSCLC",
        "predicted": "NSCLC",
        "correct": True
    },
    {
        "sentence": "Robot-assisted thoracic surgery versus video-assisted thoracic surgery for lung lobectomy or segmentectomy in patients with nonsmall cell lung cancer: a meta-analysis.",
        "label": "NSCLC",
        "predicted": "NSCLC",
        "correct": True
    },
    {
        "sentence": "Stereotactic ablative radiotherapy versus standard radiotherapy in stage 1 nonsmall cell lung cancer (TROG 09.02 CHISEL): a phase 3, open label, randomised controlled trial.",
        "label": "NSCLC",
        "predicted": "NSCLC",
        "correct": True
    },
    {
        "sentence": "If disease flare occurs, restart TKI.",
        "label": "NSCLC",
        "predicted": "NSCLC",
        "correct": True
    },
    {
        "sentence": "If FDG PET/CT is not available, bone scan may be used to identify metastases.",
        "label": "SCLC",
        "predicted": "NSCLC",
        "correct": False
    }
]

df_samples = pd.DataFrame(sample_data)

with st.expander("🔍 See Sample Predictions Table"):
    st.dataframe(df_samples, use_container_width=True)

# Let the user pick an example or paste their own
sample_sentences = df_samples["sentence"].tolist()
selected_example = st.selectbox(
    "Or pick a sample sentence to test:",
    [""] + sample_sentences,
    format_func=lambda x: "Choose..." if x == "" else (x[:70] + "..." if len(x) > 70 else x),
    help="Selecting an example will populate the text area below."
)

if selected_example and selected_example != "Choose...":
    input_text = selected_example
else:
    input_text = ""

input_text = st.text_area(
    "NCCN guideline text:",
    value=input_text,
    height=150,
    key="nccn_input"
)

if st.button("Analyze"):
    if not input_text.strip():
        st.warning("Please enter or select an NCCN guideline sentence.")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = predict_nccn(input_text)
                st.success("Model Output:")
                st.write(result)
            except Exception as e:
                st.error(f"Error running prediction: {e}")

st.caption("Developed by Evan Moh for AIPI-540 Deep Learning · Duke University")

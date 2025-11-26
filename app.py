import os
import traceback
import joblib
import pandas as pd
import streamlit as st



try:
    import google.genai as genai
    HAVE_GENAI = True
except:
    HAVE_GENAI = False

GENAI_API_KEY = os.getenv("GENAI_API_KEY", "AIzaSyAxcv9vIvO0MfPLjzZ290JDBP3CdcOKA6w")
if GENAI_API_KEY and HAVE_GENAI:
    try:
        client = genai.Client(api_key=GENAI_API_KEY)
        ai_enabled = True
    except Exception as e:
        client = None
        ai_enabled = False
        st.sidebar.error(f"AI setup failed: {e}")
else:
    client = None
    ai_enabled = False



# ---------------- Initialize session state ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "prediction_message" not in st.session_state:
    st.session_state.prediction_message = ""

# ---------------- Page config ----------------
st.set_page_config(page_title="❤️ Heart Attack Prediction", layout="wide")
st.title("❤️ Heart Attack Prediction App")

# ---------------- Model & feature definitions ----------------
MODEL_PATH = "random_forest_heart_attack.pkl"

# These must match exactly the features used during model training
required_columns = [
    'Sex', 'PhysicalHealthDays', 'MentalHealthDays', 'PhysicalActivities',
    'RemovedTeeth', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer',
    'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
    'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
    'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing',
    'DifficultyErrands', 'SmokerStatus', 'ECigaretteUsage', 'ChestScan',
    'AgeCategory', 'HeightInMeters', 'WeightInKilograms', 'BMI',
    'AlcoholDrinkers', 'FluVaxLast12', 'PneumoVaxEver'
]

one_zero_columns = {
    'Sex', 'PhysicalActivities', 'RemovedTeeth', 'HadAngina', 'HadStroke',
    'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder',
    'HadKidneyDisease', 'HadArthritis', 'HadDiabetes',
    'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating',
    'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands',
    'SmokerStatus', 'ECigaretteUsage', 'ChestScan', 'AlcoholDrinkers',
    'FluVaxLast12', 'PneumoVaxEver'
}

numeric_columns = {
    'PhysicalHealthDays', 'MentalHealthDays', 'AgeCategory', 'HeightInMeters',
    'WeightInKilograms', 'BMI'
}

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def load_csv_bytes(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_resource(show_spinner=False)
def load_model(path):
    try:
        mdl = joblib.load(path)
        return mdl, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, tb

# ---------------- Load model ----------------
model, model_load_error = load_model(MODEL_PATH)
if model is None:
    st.error("⚠️ Couldn't load model from file.")
    st.code(model_load_error)
    st.info("Check that the model file exists and that scikit-learn versions are compatible.")


def clear_box():
    st.session_state.input_box = ""
def rerun():
    st.session_state._rerun = True

# ----------------- Sidebar AI Chatbot (WhatsApp-style) -----------------
with st.sidebar:
    st.header("💬 AI Health Assistant")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "ai_input_box" not in st.session_state:
        st.session_state.ai_input_box = "" 

    # Function to handle sending message
    def send_message():
        user_msg = st.session_state.ai_input_box.strip()
        if not user_msg:
            return
        if ai_enabled:
            with st.spinner("🤖 AI is thinking..."):
                try:
                    system_prompt = (
                        "You are a professional AI health assistant. "
                        "Answer strictly about medical, health, and lifestyle topics."
                    )
                    full_prompt = f"{system_prompt}\n\nUser question: {user_msg}"
                    response = client.models.generate_content(
                        model="models/gemini-2.5-flash",
                        contents=full_prompt
                    )
                    try:
                        ai_text = response.text
                    except:
                        ai_text = "".join([p.text for p in response.candidates[0].content.parts])
                except Exception:
                    ai_text = "⚠️ AI request failed or returned no text."
        else:
            ai_text = "⚠️ AI assistant is disabled."

        st.session_state.chat_history.append([user_msg, ai_text])
        st.session_state.ai_input_box = ""  # clear input

    # Fixed suggested questions
    st.markdown("**💡 Quick Questions:**")
    fixed_questions = [
        "What are the symptoms of heart disease?",
        "How can I reduce my risk of a heart attack?",
        "What is a healthy cholesterol level?",
        "Explain different types of heart attacks",
        "How often should I get a heart checkup?"
    ]

    # When user clicks a suggested question
    for i, q in enumerate(fixed_questions):
        if st.button(q, key=f"suggest_{i}"):
            st.session_state.ai_input_box = q
            send_message()
            rerun()  

    # User input box with on_change (Enter triggers send_message)
    st.text_input(
        "Type your question:", 
        key="ai_input_box",
        value=st.session_state.ai_input_box,
        placeholder="E.g., What is a healthy BMI?",
        on_change=send_message
    )

    # Display chat history in WhatsApp style (latest first)
    st.markdown("---")
    for user_msg, ai_msg in reversed(st.session_state.chat_history):
        st.markdown(f"""
        <div style='background:#DCF8C6; padding:8px 12px; border-radius:12px; margin-bottom:5px;'>
            <strong> You:</strong> {user_msg}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:#FFF; padding:8px 12px; border-radius:12px; margin-bottom:10px;'>
            <strong>🤖 AI:</strong> {ai_msg}
        </div>
        """, unsafe_allow_html=True)



# ---------------- Tabs ----------------
tab1, tab2 = st.tabs(["Single prediction", "Upload CSV for batch prediction"])

# ---------------- Single prediction ----------------
with tab1:
    st.subheader("Single patient input")
    st.write("Fill the patient data below. Fields are kept in English to match the model feature names.")

    # Build data dict
    data = {}

    # Disabilities master question -> show multiselect only if Yes
    st.markdown("**👤 Disabilities (optional)**")
    has_disability = st.radio("Does the patient have any disability?", ("No", "Yes"), horizontal=True, key="has_dis")
    # default zeros for disability-related features
    disability_keys = ["DeafOrHardOfHearing", "BlindOrVisionDifficulty", "DifficultyConcentrating",
                       "DifficultyWalking", "DifficultyDressingBathing", "DifficultyErrands"]
    if has_disability == "Yes":
        disability_labels = [
            ("DeafOrHardOfHearing", "Hearing difficulty"),
            ("BlindOrVisionDifficulty", "Vision difficulty"),
            ("DifficultyConcentrating", "Difficulty concentrating"),
            ("DifficultyWalking", "Difficulty walking"),
            ("DifficultyDressingBathing", "Difficulty with dressing/bathing"),
            ("DifficultyErrands", "Difficulty doing errands")
        ]
        options = [label for (_, label) in disability_labels] + ["Other"]
        sel = st.multiselect("Select disability types (if any):", options, default=[])
        for key, label in disability_labels:
            data[key] = 1 if label in sel else 0
        if "Other" in sel:
            st.text_input("Specify other disability (optional):", key="other_disability")
    else:
        for k in disability_keys:
            data[k] = 0

    # Core inputs in 3 columns
    st.markdown("**🩺 Core Medical Inputs**")
    c1, c2, c3 = st.columns(3)
    with c1:
        data["Sex"] = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male", key="sex")
        data["AgeCategory"] = st.number_input("Age (years)", min_value=0, max_value=120, value=40, key="age")
        data["PhysicalHealthDays"] = st.number_input("Physical health days (last 30 days)", min_value=0, value=0, key="phd")
        data["MentalHealthDays"] = st.number_input("Mental health days (last 30 days)", min_value=0, value=0, key="mhd")
    with c2:
        data["HeightInMeters"] = st.number_input("Height (meters)", min_value=0.0, value=1.70, format="%.2f", key="h")
        data["WeightInKilograms"] = st.number_input("Weight (kg)", min_value=0.0, value=70.0, format="%.1f", key="w")
        data["BMI"] = st.number_input("BMI (if known)", min_value=0.0, value=24.2, format="%.2f", key="bmi")
    with c3:
        data["PhysicalActivities"] = st.selectbox("Physical Activities", [0,1], format_func=lambda x: "No" if x==0 else "Yes", key="pa")
        data["SmokerStatus"] = st.selectbox("Smoker", [0,1], format_func=lambda x: "No" if x==0 else "Yes", key="smoke")
        data["ECigaretteUsage"] = st.selectbox("E-Cigarette usage", [0,1], format_func=lambda x: "No" if x==0 else "Yes", key="ecig")

    # Clinical flags
    st.markdown("**📋 Clinical History & Flags**")

    def yn_radio(question, key):
        return 1 if st.radio(question, ["No", "Yes"], horizontal=True, key=key) == "Yes" else 0

    c1, c2, c3 = st.columns(3)

    with c1:
        data["HadAngina"] = yn_radio("Have you ever experienced chest pain or pressure (Angina)?", "angina_q")
        data["HadStroke"] = yn_radio("Have you ever had a stroke?", "stroke_q")
        data["HadAsthma"] = yn_radio("Do you have asthma or breathing problems?", "asthma_q")

    with c2:
        data["HadSkinCancer"] = yn_radio("Have you been diagnosed with skin cancer?", "skin_q")
        data["HadCOPD"] = yn_radio("Do you have chronic lung diseases such as COPD?", "copd_q")
        data["HadDepressiveDisorder"] = yn_radio("Have you been diagnosed with depressive disorder?", "dep_q")

    with c3:
        data["HadKidneyDisease"] = yn_radio("Do you have any kidney-related diseases?", "kidney_q")
        data["HadArthritis"] = yn_radio("Do you suffer from arthritis or chronic joint pain?", "arth_q")
        data["HadDiabetes"] = yn_radio("Do you have diabetes (Type 1 or Type 2)?", "diab_q")

    # Scans & lifestyle
    st.markdown("**🔬 Scans, Vaccines & Lifestyle**")
    sl_cols = st.columns(3)
    sl_keys = ["ChestScan","RemovedTeeth","AlcoholDrinkers","FluVaxLast12","PneumoVaxEver"]
    for i, k in enumerate(sl_keys):
        with sl_cols[i % 3]:
            data[k] = st.selectbox(k + " (No/Yes)", [0,1], format_func=lambda x: "No" if x==0 else "Yes", key=f"sl_{k}")

    # Ensure all required columns are present in data dict
    for col in required_columns:
        if col not in data:
            if col in one_zero_columns:
                data[col] = 0
            elif col in numeric_columns:
                data[col] = 0.0
            else:
                data[col] = 0

    # Build ordered DataFrame to pass to model
    input_df = pd.DataFrame([{c: data.get(c, 0) for c in required_columns}])

    st.markdown("### Input preview (will be passed to model):")
    st.dataframe(input_df)

 # ----------------- Prediction & AI -----------------
prediction = 0
prob = None
ai_input = ""

if st.button("Predict Heart Attack"):
    if model is None:
        st.error("Model not loaded.")
    else:
        input_data = pd.DataFrame([{c:data[c] for c in required_columns}])
        try:
            prediction = model.predict(input_data)[0]
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)[0][1]
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

        # Display risk
        if prediction == 1:
            st.error("❗ High Risk Detected — You may be at risk of a heart attack.")
            if prob is not None:
                st.info(f"Predicted probability of risk: {prob:.3f}")
            ai_input = f"I am at high risk of a heart attack. What lifestyle changes and precautions should I consider to reduce my risk?"
        else:
            st.success("💚 You are not at risk based on the model.")
            if prob is not None:
                st.info(f"Predicted probability of risk: {prob:.3f}")
            ai_input = f"I am not at risk of a heart attack. What lifestyle habits should I maintain to keep my heart healthy?"

        # Call AI assistant if enabled
        if ai_enabled and ai_input:
            with st.spinner("🤖 AI is thinking..."):
                try:
                    system_prompt = """
                            You are a professional AI health assistant. 
                            Answer user questions strictly about medical, health, and lifestyle topics.
                            Always provide accurate, evidence-based information.
                            Do NOT give advice outside of health or medical context.
                                """

                    user_question = ai_input 

                    full_prompt = f"{system_prompt}\n\nUser question: {user_question}"

                    response = client.models.generate_content(model="models/gemini-2.5-flash",contents=full_prompt)
                    ai_text = response.text

                    try:
                        ai_text = response.text
                    except:
                        ai_text = "".join([p.text for p in response.candidates[0].content.parts])
                except Exception:
                    ai_text = "AI request failed or returned no text."
            st.markdown(f"<span style='color:white; padding:10px;'>**🤖 AI Assistant:** {ai_text}</span>", unsafe_allow_html=True)


# ---------------- Batch CSV prediction ----------------
with tab2:
    st.subheader("Batch prediction (CSV)")
    st.write("Upload a CSV whose headers *exactly match* the model feature names (shown below).")
    with st.expander("Show required feature names", expanded=False):
        st.write(required_columns)

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="batch_upload")
    if uploaded_file is not None:
        try:
            df = load_csv_bytes(uploaded_file)
            st.success("File uploaded.")
        except Exception as e:
            st.error("Failed to read CSV file.")
            st.exception(e)
            df = None

        if df is not None:
            st.write("Uploaded file columns:")
            st.write(df.columns.tolist())

            # Check missing/extra
            missing_cols = [c for c in required_columns if c not in df.columns]
            extra_cols = [c for c in df.columns if c not in required_columns]

            if missing_cols:
                st.error("⚠️ Missing required columns (you must add them or fix the CSV):")
                st.write(missing_cols)
            if extra_cols:
                st.warning("⚠️ Extra columns detected (these will be ignored for prediction):")
                st.write(extra_cols)

            # If missing columns exist, do not allow prediction
            if missing_cols:
                st.info("Please fix CSV and re-upload. (Missing columns prevent prediction.)")
            else:
                # build input_batch with exact order (ignore extras)
                input_batch = df[required_columns].copy()

                # coerce numeric columns
                for nc in numeric_columns:
                    input_batch[nc] = pd.to_numeric(input_batch[nc], errors="coerce").fillna(0.0)
                # ensure 0/1 columns are ints
                for c in one_zero_columns:
                    if c in input_batch.columns:
                        input_batch[c] = input_batch[c].fillna(0).astype(int)

                st.markdown("Preview of data that will be passed to model (first 5 rows):")
                st.dataframe(input_batch.head())

                if st.button("Predict Heart Attack for Uploaded CSV", key="predict_csv"):
                    if model is None:
                        st.error("Model not loaded — cannot predict.")
                    else:
                        try:
                            preds = model.predict(input_batch)
                            result_df = input_batch.copy()
                            result_df['HeartAttackPrediction'] = preds
                            if hasattr(model, "predict_proba"):
                                probs = model.predict_proba(input_batch)[:, 1]
                                result_df['RiskProbability'] = probs
                            st.success("✅ Predictions completed")
                            st.dataframe(result_df)
                            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                            st.download_button("⬇️ Download predictions CSV",
                                               data=csv_bytes,
                                               file_name="heart_attack_predictions.csv",
                                               mime="text/csv",
                                               key="download_predictions")
                        except Exception as e:
                            st.error("Prediction failed. See details:")
                            st.exception(e)

# ---------------- Footer ----------------
st.markdown(
    """
    <style>
    .footer {position:fixed; left:0; bottom:0; width:100%; background-color:#0E1117;
            color:white; text-align:center; padding:10px; font-size:14px; border-top:1px solid #333;}
    .footer a{color:#4CAF50;}
    </style>
    <div class="footer">👨‍💻 Developed by our team</div>
    """,
    unsafe_allow_html=True)

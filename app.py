"""
app.py
Streamlit Web Application for SIH 1697:
"Groundwater Knowledge Chatbot with Data Retrieval"
End-to-end prototype combining TF-IDF + Multinomial Naive Bayes for Intent Classification
and District-level Data Retrieval from Supplied Government Groundwater Dataset (Telangana).
"""

import os
import re
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Groundwater Knowledge Chatbot",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .answer-card {
        background: #EFF6FF;
        border-radius: 8px;
        padding: 1.2rem;
        border: 1px solid #BFDBFE;
        color: #1E3A8A;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    .metric-container {
        display: flex;
        gap: 15px;
        margin-bottom: 10px;
    }
    .metric-badge {
        background-color: #F3F4F6;
        border-left: 3px solid #2563EB;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data & Model Loaders
# ---------------------------------------------------------
def locate_file(filenames):
    for f in filenames:
        if os.path.exists(f):
            return f
    return filenames[0]

DATA_PATH = locate_file([
    os.path.join("data", "tg_gw_2025 (1).csv"),
    os.path.join("..", "data", "tg_gw_2025 (1).csv"),
    os.path.join("tg_gw_2025 (1).csv"),
    os.path.join("..", "tg_gw_2025 (1).csv"),
    os.path.join("groundwater-chatbot", "data", "tg_gw_2025 (1).csv")
])

MODEL_PATH = locate_file([
    os.path.join("models", "groundwater_nb_pipeline.pkl"),
    os.path.join("..", "models", "groundwater_nb_pipeline.pkl"),
    os.path.join("groundwater-chatbot", "models", "groundwater_nb_pipeline.pkl")
])


@st.cache_resource
def load_resources():
    # 1. Load trained ML pipeline (TF-IDF + MultinomialNB)
    pipeline = joblib.load(MODEL_PATH)
    
    # 2. Load Government groundwater dataset
    df = pd.read_csv(DATA_PATH)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # 3. Filter district records (Telangana has 33 official districts)
    district_df = df[df["Sl. No"].notnull()].copy()
    if district_df.empty:
        district_df = df[~df["Name of District"].str.startswith("Total")].copy()
        
    return pipeline, df, district_df

try:
    pipeline, df_full, district_df = load_resources()
except Exception as e:
    st.error(f"Error loading system resources: {e}")
    st.stop()


# ---------------------------------------------------------
# District & Location Detection Logic
# ---------------------------------------------------------
DISTRICT_LIST = district_df["Name of District"].tolist()

ALIASES = {
    "bhadradri": "Bhadradri Kothagudem",
    "kothagudem": "Bhadradri Kothagudem",
    "bhupalapally": "Jayashankar Bhupalapally",
    "gadwal": "Jogulamba Gadwal",
    "asifabad": "Komarambheem Asifabad",
    "komaram bheem": "Komarambheem Asifabad",
    "siricilla": "Rajanna Siricilla",
    "peddapalli": "Peddapalle",
    "bhuvanagiri": "Yadadri Bhuvanagiri",
    "bhongir": "Yadadri Bhuvanagiri",
    "malkajgiri": "Medchal Malkajgiri",
    "medchal": "Medchal Malkajgiri"
}

STOP_WORDS_LOC = {
    "groundwater", "water", "extraction", "recharge", "availability", "stage",
    "future", "use", "district", "districts", "telangana", "state", "percentage",
    "resource", "resources", "irrigation", "domestic", "industrial", "total",
    "annual", "extractable", "highest", "lowest", "least", "most", "compare"
}

def detect_districts(query: str):
    """
    Extracts all known Telangana districts and identifies if an unrecognized
    district/location was mentioned.
    """
    q_norm = query.lower().replace("-", " ")
    found = []
    
    # 1. Match known districts (longer names prioritized)
    for dist in sorted(DISTRICT_LIST, key=len, reverse=True):
        if dist.lower() in q_norm and dist not in found:
            found.append(dist)
            
    # 2. Check aliases
    for alias, official in ALIASES.items():
        if alias in q_norm and official not in found:
            found.append(official)
            
    if found:
        return found, None
        
    # 3. Detect unrecognized location pattern (e.g., 'in Chennai', 'for Delhi')
    matches = re.findall(r"\b(?:in|for|of|at)\s+([A-Za-z]+)", query, re.IGNORECASE)
    for m in matches:
        if m.lower() not in STOP_WORDS_LOC:
            return [], m.title()
            
    return [], None


# ---------------------------------------------------------
# Dynamic Retrieval Logic (Strict Zero-Hallucination)
# ---------------------------------------------------------
def determine_comparison_metric(q_lower: str):
    if "recharge" in q_lower:
        return "Total annual groundwater recharge", "Total annual groundwater recharge", "Ham"
    elif "stage" in q_lower or "%" in q_lower or "percentage" in q_lower:
        return "Stage of GW extraction (%)", "Stage of groundwater extraction", "%"
    elif "future" in q_lower:
        return "Net GW availability for future", "Net groundwater availability for future use", "Ham"
    elif "availability" in q_lower or "available" in q_lower or "resource" in q_lower:
        return "Annual Extractable Groundwater Resource", "Annual extractable groundwater resource", "Ham"
    elif "irrigation" in q_lower:
        return "Irrigation - Annual extraction", "Irrigation groundwater extraction", "Ham"
    elif "industrial" in q_lower:
        return "Industrial - Annual extraction", "Industrial groundwater extraction", "Ham"
    elif "domestic" in q_lower:
        return "Domestic - Annual Extraction", "Domestic groundwater extraction", "Ham"
    else:
        return "Total Annual Extraction", "Total annual groundwater extraction", "Ham"


def determine_extraction_column(q_lower: str):
    if "industrial" in q_lower:
        return "Industrial - Annual extraction", "industrial groundwater extraction", "Ham"
    elif "domestic" in q_lower:
        return "Domestic - Annual Extraction", "domestic groundwater extraction", "Ham"
    elif "irrigation" in q_lower:
        return "Irrigation - Annual extraction", "irrigation groundwater extraction", "Ham"
    else:
        return "Total Annual Extraction", "total annual groundwater extraction", "Ham"


def process_query(query: str):
    q_clean = query.strip()
    if not q_clean:
        return {
            "success": False,
            "error": "Please enter a valid groundwater query."
        }

    # 1. Predict question intent using Naive Bayes pipeline
    intent = pipeline.predict([q_clean])[0]
    q_lower = q_clean.lower()

    # 2. Extract locations
    known_districts, unknown_district = detect_districts(q_clean)

    # 3. Handle Comparison Questions
    if intent == "comparison":
        col, metric_label, unit = determine_comparison_metric(q_lower)

        # Multi-district comparison (e.g. Compare recharge in Adilabad and Jagtial)
        if len(known_districts) >= 2:
            d1, d2 = known_districts[0], known_districts[1]
            row1 = district_df[district_df["Name of District"] == d1]
            row2 = district_df[district_df["Name of District"] == d2]
            
            val1 = row1[col].values[0]
            val2 = row2[col].values[0]
            
            unit_str = f" {unit}" if unit != "%" else "%"
            answer = (
                f"{d1} has a {metric_label.lower()} of {val1:,.2f}{unit_str}, while "
                f"{d2} has {val2:,.2f}{unit_str}."
            )
            rows = pd.concat([row1, row2])
            
            return {
                "success": True,
                "intent": intent,
                "location": f"{d1}, {d2}",
                "retrieved_metric": col,
                "retrieved_value": f"{d1}: {val1:,.2f}{unit_str}, {d2}: {val2:,.2f}{unit_str}",
                "answer": answer,
                "row": rows
            }

        # Statewide Extremum comparison (highest / lowest)
        is_lowest = any(w in q_lower for w in ["lowest", "least", "minimum", "min"])
        target_idx = district_df[col].idxmin() if is_lowest else district_df[col].idxmax()
        d_name = district_df.loc[target_idx, "Name of District"]
        val = district_df.loc[target_idx, col]
        comp_word = "lowest" if is_lowest else "highest"
        unit_str = f" {unit}" if unit != "%" else "%"

        answer = f"{d_name} has the {comp_word} {metric_label.lower()} of {val:,.2f}{unit_str}."
        row = district_df.loc[[target_idx]]

        return {
            "success": True,
            "intent": intent,
            "location": d_name,
            "retrieved_metric": col,
            "retrieved_value": f"{val:,.2f}{unit_str}",
            "answer": answer,
            "row": row
        }

    # 4. Handle Unrecognized / Non-Telangana District (e.g. Chennai)
    if unknown_district:
        return {
            "success": False,
            "intent": intent,
            "location": unknown_district,
            "error": "That information is not available in the current government dataset."
        }

    # 5. Handle Missing District
    if not known_districts:
        return {
            "success": False,
            "intent": intent,
            "location": None,
            "error": "Please mention a Telangana district such as Hyderabad, Karimnagar or Adilabad."
        }

    district = known_districts[0]
    row = district_df[district_df["Name of District"] == district]
    if row.empty:
        return {
            "success": False,
            "intent": intent,
            "location": district,
            "error": "That information is not available in the current government dataset."
        }

    # 6. Intent to Metric Mapping
    if intent == "extraction":
        col, metric_label, unit = determine_extraction_column(q_lower)
    elif intent == "recharge":
        col, metric_label, unit = "Total annual groundwater recharge", "total annual groundwater recharge", "Ham"
    elif intent == "availability":
        col, metric_label, unit = "Annual Extractable Groundwater Resource", "annual extractable groundwater resource", "Ham"
    elif intent == "future_availability":
        col, metric_label, unit = "Net GW availability for future", "net groundwater availability for future use", "Ham"
    elif intent == "stage":
        col, metric_label, unit = "Stage of GW extraction (%)", "groundwater extraction stage", "%"
    else:
        return {
            "success": False,
            "intent": intent,
            "location": district,
            "error": "That information is not available in the current government dataset."
        }

    val = row[col].values[0]
    unit_str = f" {unit}" if unit != "%" else "%"
    answer = f"{district} has a {metric_label} of {val:,.2f}{unit_str}."

    return {
        "success": True,
        "intent": intent,
        "location": district,
        "retrieved_metric": col,
        "retrieved_value": f"{val:,.2f}{unit_str}",
        "answer": answer,
        "row": row
    }


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("📌 Model Information")
    st.markdown("""
    **Model:**  
    Multinomial Naive Bayes  

    **Text representation:**  
    TF-IDF  

    **Training examples:**  
    47  

    **Intent classes:**  
    6 (`recharge`, `availability`, `extraction`, `stage`, `comparison`, `future_availability`)  

    **Evidence source:**  
    Supplied Government Groundwater Dataset (`tg_gw_2025 (1).csv`)  
    State: **Telangana** (33 Districts)
    """)
    st.divider()
    st.info("💡 **Grounded Retrieval**: All numerical answers are dynamically retrieved from the government CSV using pandas operations. No hallucinations or LLM generation.")


# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown('<div class="main-title">💧 Groundwater Knowledge Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Government-data-based groundwater query prototype</div>', unsafe_allow_html=True)

# Example questions helper
example_questions = [
    "What is the groundwater extraction in Hyderabad?",
    "What is the groundwater recharge in Hyderabad?",
    "What is the industrial groundwater extraction in Hyderabad?",
    "What is the domestic groundwater extraction in Hyderabad?",
    "What is the irrigation groundwater extraction in Nalgonda?",
    "What is the groundwater availability in Adilabad?",
    "What is the extraction stage in Jagtial?",
    "Which district has the highest groundwater extraction?",
    "Which district has the lowest groundwater extraction?",
    "Which district has the highest groundwater recharge?",
    "Which district has the highest extraction stage?",
    "Which district has the highest future groundwater availability?",
    "What is the future groundwater availability in Karimnagar?",
    "Compare groundwater recharge in Adilabad and Jagtial.",
    "What is groundwater extraction in Chennai?",
    "What is the groundwater extraction?"
]

st.markdown("##### 🔍 Try Example Questions:")
cols = st.columns(4)
for i, ex in enumerate(example_questions):
    if cols[i % 4].button(ex, key=f"ex_btn_{i}", width="stretch"):
        st.session_state["user_query_input"] = ex

default_query = st.session_state.get("user_query_input", "")

# Search Input
user_query = st.text_input("Ask your groundwater question:", value=default_query, placeholder="e.g. What is the groundwater extraction in Hyderabad?")
get_answer_btn = st.button("Get Answer", type="primary")

if get_answer_btn or (user_query and user_query != default_query):
    if not user_query.strip():
        st.warning("Please enter a question about Telangana groundwater.")
    else:
        res = process_query(user_query)
        st.markdown("---")
        
        if not res["success"]:
            # Display Prediction metadata if intent was predicted
            if "intent" in res and res["intent"]:
                st.subheader("Prediction")
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.write(f"**Query Intent:** `{res['intent']}`")
                with col_p2:
                    st.write(f"**Location:** `{res['location'] if res['location'] else 'None detected'}`")
                st.markdown("---")
            st.warning(f"⚠️ {res['error']}")
        else:
            # 1. Prediction Block
            st.subheader("Prediction")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.write(f"**Query Intent:** `{res['intent']}`")
            with col_p2:
                st.write(f"**Location:** `{res['location']}`")

            st.markdown("---")

            # 2. Answer Block
            st.subheader("Answer")
            st.markdown(f'<div class="answer-card">✅ {res["answer"]}</div>', unsafe_allow_html=True)

            # Metadata Badges: Metric & Value
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.info(f"📊 **Retrieved Metric:** `{res['retrieved_metric']}`")
            with col_m2:
                st.success(f"📈 **Retrieved Value:** `{res['retrieved_value']}`")

            st.markdown("---")

            # 3. Evidence Block
            st.subheader("Evidence")
            st.markdown("""
            **Source:**  
            Supplied Government Groundwater Dataset  

            **Dataset:**  
            `tg_gw_2025 (1).csv` (Telangana State)
            """)

            # 4. Expandable Raw Data
            with st.expander("View Evidence (Retrieved Government CSV Row)"):
                st.dataframe(res["row"], width="stretch")


# ---------------------------------------------------------
# Analytics Section
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Groundwater Analytics")

tab1, tab2, tab3 = st.tabs([
    "Groundwater Extraction by District",
    "Extraction Stage by District (%)",
    "Groundwater Recharge by District"
])

with tab1:
    st.markdown("**Total Annual Groundwater Extraction across 33 Districts (Ham)**")
    chart_data1 = district_df[["Name of District", "Total Annual Extraction"]].sort_values(
        by="Total Annual Extraction", ascending=False
    ).set_index("Name of District")
    st.bar_chart(chart_data1, color="#2563EB")

with tab2:
    st.markdown("**Stage of Groundwater Extraction (%) across Districts** (Safe < 70%, Semi-Critical 70-90%, Critical > 90%)")
    chart_data2 = district_df[["Name of District", "Stage of GW extraction (%)"]].sort_values(
        by="Stage of GW extraction (%)", ascending=False
    ).set_index("Name of District")
    st.bar_chart(chart_data2, color="#DC2626")

with tab3:
    st.markdown("**Total Annual Groundwater Recharge across 33 Districts (Ham)**")
    chart_data3 = district_df[["Name of District", "Total annual groundwater recharge"]].sort_values(
        by="Total annual groundwater recharge", ascending=False
    ).set_index("Name of District")
    st.bar_chart(chart_data3, color="#059669")

# 💧 Groundwater Knowledge Chatbot with Data Retrieval (SIH 1697 Prototype)

An end-to-end, lightweight prototype for the Smart India Hackathon (SIH 1697) problem: **Groundwater Knowledge Chatbot with Data Retrieval**.

This application maps user questions into machine learning intents using **TF-IDF + Multinomial Naive Bayes**, extracts district entities, and performs zero-hallucination grounded data retrieval directly from the official **Telangana Government Groundwater Dataset**.

---

## 📋 Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Datasets](#2-datasets)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
5. [Text Representation (TF-IDF)](#5-text-representation-tf-idf)
6. [Intent Classification (Naive Bayes)](#6-intent-classification-naive-bayes)
7. [Location Detection & Grounded Retrieval](#7-location-detection--grounded-retrieval)
8. [Streamlit Web Application](#8-streamlit-web-application)
9. [How to Run](#9-how-to-run)
10. [Example Test Queries](#10-example-test-queries)

---

## 1. Problem Statement
Groundwater resource planning and public awareness require quick, transparent access to verified hydrological data. Large Language Models often hallucinate numerical values when answering specific statistical questions. This prototype solves this challenge using a deterministic, explainable pipeline:
- Predict user intent via a lightweight **TF-IDF + Multinomial Naive Bayes** classifier.
- Detect Telangana district entities from user queries.
- Retrieve exact numbers directly from government CSV tables without numerical fabrication.
- Present predictions, grounded answers, evidence sources, and interactive analytics directly from the Supplied Government Groundwater Dataset.

---

## 2. Datasets
The prototype utilizes two datasets located in `data/`:
1. `tg_gw_2025 (1).csv`: Official Telangana State groundwater assessment dataset containing 33 districts across 16 hydrological parameters (Total Recharge, Annual Extraction, Stage of Extraction, Future Availability, Irrigation/Domestic use, etc.).
   > *Note:* This dataset represents **Telangana districts**, not Tamil Nadu.
2. `groundwater_query_intent_dataset.csv`: 47 curated groundwater query examples labeled across 6 user intents:
   - `recharge`
   - `availability`
   - `extraction`
   - `stage`
   - `comparison`
   - `future_availability`

---

## 3. Data Preprocessing
- **Whitespace normalization**: Leading and trailing whitespaces stripped across all string and district columns.
- **District filtering**: Separate 33 official district records from state total summary rows (`Total(Ham)` and `Total(Bcm)`) to ensure accurate district comparisons.
- **Type safety**: Numerical columns verified and formatted for unit consistency (`Ham` and `%`).
- **Missing values**: Verified that all 33 district records contain zero null groundwater values.

---

## 4. Exploratory Data Analysis (EDA)
The script `analysis.py` performs descriptive statistical analysis and saves 3 visualizations into `outputs/`:
- `outputs/extraction_by_district.png`: Bar chart of Total Annual Extraction by district.
- `outputs/extraction_stage_by_district.png`: Color-coded extraction stage highlighting Safe (<70%), Semi-Critical (70–90%), and Critical (>90%) zones.
- `outputs/recharge_by_district.png`: Total Annual Groundwater Recharge across Telangana districts.

---

## 5. Text Representation (TF-IDF)
`TfidfVectorizer` converts natural language user questions into term frequency-inverse document frequency vector representations. Sublinear TF and word-level n-grams capture question keywords (`recharge`, `extraction`, `available`, `stage`, `highest`, `future`) while remaining lightweight and fast.

---

## 6. Intent Classification (Naive Bayes)
A `MultinomialNB` pipeline classifies questions into one of the 6 core intents:
- Evaluated on a stratified 20% test split (`test_size=0.2, random_state=42, stratify=y`).
- The complete pipeline (`TfidfVectorizer + MultinomialNB`) is serialized using `joblib` into `models/groundwater_nb_pipeline.pkl`.

---

## 7. Location Detection & Grounded Retrieval
- **Location Detection**: Case-insensitive district entity matcher across all 33 official Telangana districts, with alias support (e.g., `Bhadradri Kothagudem`, `Medchal Malkajgiri`, `Jogulamba Gadwal`).
- **Intent Mapping**:
  - `recharge` $\rightarrow$ `Total annual groundwater recharge`
  - `availability` $\rightarrow$ `Annual Extractable Groundwater Resource`
  - `extraction` $\rightarrow$ `Total Annual Extraction`
  - `future_availability` $\rightarrow$ `Net GW availability for future`
  - `stage` $\rightarrow$ `Stage of GW extraction (%)`
  - `comparison` $\rightarrow$ Dynamic `pandas` analytical max/min query (e.g., highest recharge, lowest extraction).
- **Error Handling**: Graceful fallback when a district is not mentioned, prompting the user to name a Telangana district.

---

## 8. Streamlit Web Application
`app.py` delivers a clean, responsive UI:
- **Prediction Section**: Displays classified Query Intent and detected District.
- **Answer Section**: Shows grounded factual response.
- **Evidence Section**: Displays Supplied Government Groundwater Dataset source, exact retrieved metric, retrieved value, and an expandable raw CSV row viewer (`st.expander("View Evidence")`).
- **Groundwater Analytics**: Interactive bar charts for Extraction, Extraction Stage, and Recharge.
- **Sidebar**: Complete model architecture and configuration metadata.

---

## 9. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Exploratory Data Analysis & Generate Visualizations
```bash
python analysis.py
```

### Step 3: Train & Save Naive Bayes Intent Pipeline
```bash
python train.py
```

### Step 4: Launch Streamlit Chatbot Application
```bash
streamlit run app.py
```

---

## 10. Example Test Queries
| Query | Predicted Intent | Location | Retrieved Value / Status |
|---|---|---|---|
| "What is the groundwater extraction in Hyderabad?" | `extraction` | Hyderabad | 6,061.83 Ham |
| "What is the groundwater recharge in Karimnagar?" | `recharge` | Karimnagar | 93,807.34 Ham |
| "What is the groundwater availability in Adilabad?" | `availability` | Adilabad | 51,020.35 Ham |
| "What is the extraction stage in Jagtial?" | `stage` | Jagtial | 35.88% |
| "Which district has the highest groundwater extraction?" | `comparison` | Nalgonda | 63,153.93 Ham |
| "What is the future groundwater availability in Hyderabad?" | `future_availability` | Hyderabad | 1,057.75 Ham |
| "What is the groundwater extraction?" | `extraction` | None | Friendly prompt for district |

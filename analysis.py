"""
analysis.py
Performs Exploratory Data Analysis (EDA) on the Telangana Government Groundwater Dataset.
Inspects shape, columns, dtypes, missing values, duplicates, descriptive statistics,
and generates key visualizations saved to the outputs/ directory.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd


def get_data_path():
    """Find the path to the government groundwater dataset."""
    possible_paths = [
        os.path.join("data", "tg_gw_2025 (1).csv"),
        os.path.join("..", "data", "tg_gw_2025 (1).csv"),
        os.path.join("tg_gw_2025 (1).csv"),
        os.path.join("..", "tg_gw_2025 (1).csv"),
        os.path.join("groundwater-chatbot", "data", "tg_gw_2025 (1).csv"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find 'tg_gw_2025 (1).csv' dataset file.")


def get_output_dir():
    """Ensure and return outputs directory."""
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main():
    data_path = get_data_path()
    print(f"Loading Government Groundwater Dataset from: {data_path}")
    df = pd.read_csv(data_path)

    print("\n" + "=" * 60)
    print("1. DATASET SHAPE")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(df.shape)

    print("\n" + "=" * 60)
    print("2. EXACT COLUMN NAMES")
    print("=" * 60)
    for idx, col in enumerate(df.columns):
        print(f"[{idx}] {col}")

    print("\n" + "=" * 60)
    print("3. DATASET HEAD")
    print("=" * 60)
    print(df.head())

    print("\n" + "=" * 60)
    print("4. DATASET INFO")
    print("=" * 60)
    df.info()

    print("\n" + "=" * 60)
    print("5. MISSING VALUES PER COLUMN")
    print("=" * 60)
    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print("6. DUPLICATE ROWS COUNT")
    print("=" * 60)
    print(f"Total duplicate rows: {df.duplicated().sum()}")

    print("\n" + "=" * 60)
    print("7. DESCRIPTIVE STATISTICS (All Columns)")
    print("=" * 60)
    print(df.describe(include="all"))

    # Clean whitespace on string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Identify district rows (excluding summary rows Total(Ham), Total(Bcm))
    district_df = df[df["Sl. No"].notnull()].copy()
    if district_df.empty:
        district_df = df[~df["Name of District"].str.startswith("Total")].copy()

    # Sort for cleaner visualizations
    dist_col = "Name of District"
    ext_col = "Total Annual Extraction"
    stage_col = "Stage of GW extraction (%)"
    rech_col = "Total annual groundwater recharge"

    out_dir = get_output_dir()
    print(f"\nSaving visualizations to: {out_dir}/")

    # Set font & aesthetic styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Extraction by District
    plt.figure(figsize=(14, 7))
    df_sorted_ext = district_df.sort_values(by=ext_col, ascending=False)
    bars1 = plt.bar(df_sorted_ext[dist_col], df_sorted_ext[ext_col], color="#2563EB", edgecolor="#1E40AF")
    plt.title("Total Annual Groundwater Extraction by District (Telangana)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("District", fontsize=11, fontweight="bold")
    plt.ylabel("Total Annual Extraction (Ham / MCM)", fontsize=11, fontweight="bold")
    plt.xticks(rotation=75, ha="right", fontsize=9)
    plt.tight_layout()
    chart1_path = os.path.join(out_dir, "extraction_by_district.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"Saved: {chart1_path}")

    # 2. Extraction Stage by District
    plt.figure(figsize=(14, 7))
    df_sorted_stage = district_df.sort_values(by=stage_col, ascending=False)
    # Color code by critical thresholds: >100 Over-exploited (red), 90-100 Critical (orange), 70-90 Semi-critical (yellow), <70 Safe (green)
    colors = []
    for val in df_sorted_stage[stage_col]:
        if val >= 90:
            colors.append("#DC2626")  # Red / Critical / Over-exploited
        elif val >= 70:
            colors.append("#F59E0B")  # Amber / Semi-critical
        else:
            colors.append("#10B981")  # Green / Safe

    plt.bar(df_sorted_stage[dist_col], df_sorted_stage[stage_col], color=colors, edgecolor="#374151")
    plt.axhline(70, color="orange", linestyle="--", linewidth=1, label="Semi-Critical (70%)")
    plt.axhline(90, color="red", linestyle="--", linewidth=1, label="Critical / High (90%)")
    plt.title("Stage of Groundwater Extraction (%) by District (Telangana)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("District", fontsize=11, fontweight="bold")
    plt.ylabel("Stage of GW Extraction (%)", fontsize=11, fontweight="bold")
    plt.xticks(rotation=75, ha="right", fontsize=9)
    plt.legend(frameon=True)
    plt.tight_layout()
    chart2_path = os.path.join(out_dir, "extraction_stage_by_district.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"Saved: {chart2_path}")

    # 3. Recharge by District
    plt.figure(figsize=(14, 7))
    df_sorted_rech = district_df.sort_values(by=rech_col, ascending=False)
    plt.bar(df_sorted_rech[dist_col], df_sorted_rech[rech_col], color="#059669", edgecolor="#047857")
    plt.title("Total Annual Groundwater Recharge by District (Telangana)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("District", fontsize=11, fontweight="bold")
    plt.ylabel("Total Annual Recharge (Ham / MCM)", fontsize=11, fontweight="bold")
    plt.xticks(rotation=75, ha="right", fontsize=9)
    plt.tight_layout()
    chart3_path = os.path.join(out_dir, "recharge_by_district.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"Saved: {chart3_path}")

    print("\nEDA and Visualizations successfully completed!")


if __name__ == "__main__":
    main()

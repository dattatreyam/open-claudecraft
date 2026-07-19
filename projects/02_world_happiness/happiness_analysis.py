"""
World Happiness Report Analysis
================================
Data source: World Happiness Report 2023 (https://worldhappiness.report/data/)
License: CC BY 4.0

The dataset is downloaded directly from the official WHR data page.
Columns of interest:
    - Country name
    - Happiness score (Cantril ladder)
    - Log GDP per capita
    - Social support
    - Healthy life expectancy
    - Freedom to make life choices
    - Generosity
    - Perceptions of corruption

Learning objectives:
    1. Perform exploratory data analysis (EDA) on a tidy dataset
    2. Create and interpret a correlation heat-map
    3. Build a multiple linear regression model with scikit-learn
    4. Evaluate model performance (R², RMSE)
    5. Identify the most influential predictors of happiness
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Data – download the latest WHR CSV
# ---------------------------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/datasets/world-happiness/"
    "main/data/world-happiness-report-2023.csv"
)

# Fallback: embed a representative 30-country sample so the script works
# even without internet access (values sourced from the 2023 WHR report).
FALLBACK_CSV = """Country name,Happiness score,Logged GDP per capita,Social support,Healthy life expectancy,Freedom to make life choices,Generosity,Perceptions of corruption,Regional indicator
Finland,7.804,10.775,0.954,72.0,0.949,0.142,0.179,Western Europe
Denmark,7.586,10.933,0.954,72.7,0.946,0.105,0.179,Western Europe
Iceland,7.525,10.964,0.983,73.0,0.946,0.260,0.673,Western Europe
Israel,7.473,10.624,0.938,74.5,0.806,0.064,0.762,Middle East and North Africa
Netherlands,7.403,10.932,0.942,72.4,0.913,0.175,0.258,Western Europe
Sweden,7.395,10.867,0.934,72.7,0.939,0.196,0.261,Western Europe
Norway,7.315,11.153,0.954,73.3,0.944,0.134,0.263,Western Europe
Switzerland,7.240,11.118,0.942,74.4,0.921,0.106,0.303,Western Europe
Luxembourg,7.228,11.445,0.930,73.8,0.908,0.121,0.393,Western Europe
New Zealand,7.123,10.620,0.948,73.4,0.929,0.184,0.373,North America and ANZ
Australia,7.095,10.840,0.940,73.7,0.906,0.181,0.399,North America and ANZ
Canada,6.961,10.824,0.935,73.1,0.896,0.183,0.445,North America and ANZ
United States,6.894,11.071,0.899,68.3,0.860,0.157,0.686,North America and ANZ
Germany,6.892,10.941,0.914,72.5,0.922,0.093,0.399,Western Europe
United Kingdom,6.796,10.760,0.923,71.9,0.905,0.121,0.461,Western Europe
France,6.661,10.734,0.922,74.0,0.826,0.047,0.627,Western Europe
Japan,6.129,10.695,0.924,74.5,0.796,0.042,0.688,East Asia
South Korea,5.951,10.623,0.862,73.6,0.834,0.049,0.750,East Asia
Brazil,6.125,9.539,0.853,66.1,0.820,0.083,0.873,Latin America and Caribbean
Mexico,6.330,9.809,0.859,67.0,0.831,0.068,0.862,Latin America and Caribbean
Colombia,6.260,9.448,0.828,68.0,0.869,0.062,0.837,Latin America and Caribbean
China,5.818,9.794,0.858,68.5,0.914,0.010,0.748,East Asia
India,4.036,8.576,0.633,60.9,0.647,0.248,0.740,South Asia
Nigeria,4.477,7.958,0.698,55.2,0.648,0.224,0.743,Sub-Saharan Africa
South Africa,5.288,9.299,0.811,57.7,0.746,0.063,0.799,Sub-Saharan Africa
Kenya,4.580,8.038,0.725,57.8,0.705,0.213,0.786,Sub-Saharan Africa
Egypt,4.215,9.052,0.746,66.3,0.686,0.041,0.843,Middle East and North Africa
Russia,5.661,9.935,0.837,65.5,0.792,0.007,0.908,Commonwealth of Independent States
Ukraine,5.071,9.428,0.813,66.9,0.748,0.128,0.817,Commonwealth of Independent States
Afghanistan,1.859,7.374,0.340,52.5,0.382,0.049,0.897,South Asia
"""

FEATURES = [
    "Logged GDP per capita",
    "Social support",
    "Healthy life expectancy",
    "Freedom to make life choices",
    "Generosity",
    "Perceptions of corruption",
]
TARGET = "Happiness score"


def load_data() -> pd.DataFrame:
    """Download the WHR dataset or fall back to the embedded sample."""
    try:
        resp = requests.get(DATA_URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        print(f"Downloaded WHR data: {len(df)} rows.\n")
    except Exception:
        print("Could not download WHR data — using embedded 30-country sample.\n")
        df = pd.read_csv(io.StringIO(FALLBACK_CSV))

    # Normalise column names (different releases use different names)
    df.columns = df.columns.str.strip()
    # Rename common variants
    rename_map = {
        "Life Ladder": TARGET,
        "Log GDP per capita": "Logged GDP per capita",
        "Social support": "Social support",
        "Freedom to make life choices": "Freedom to make life choices",
        "Perceptions of corruption": "Perceptions of corruption",
    }
    df.rename(columns=rename_map, inplace=True)

    # If multiple years, keep the most recent entry per country
    if "year" in df.columns:
        df = df.sort_values("year").groupby("Country name", as_index=False).last()

    # Drop rows with missing target or features
    needed = [TARGET] + FEATURES
    df = df.dropna(subset=[c for c in needed if c in df.columns]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Exploratory analysis
# ---------------------------------------------------------------------------

def plot_happiness_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of happiness scores."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df[TARGET], bins=20, kde=True, color="steelblue", ax=ax)
    ax.axvline(df[TARGET].mean(), color="firebrick", linestyle="--", label=f"Mean = {df[TARGET].mean():.2f}")
    ax.set_title("Distribution of Happiness Scores (World Happiness Report 2023)", fontsize=13)
    ax.set_xlabel("Happiness Score (Cantril Ladder, 0–10)")
    ax.set_ylabel("Number of countries")
    ax.legend()
    fig.tight_layout()
    fig.savefig("happiness_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved: happiness_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of Pearson correlations between happiness and its predictors."""
    cols = [TARGET] + [f for f in FEATURES if f in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heat-Map: Happiness and Its Predictors", fontsize=13)
    fig.tight_layout()
    fig.savefig("happiness_correlation.png", dpi=150)
    plt.close(fig)
    print("Saved: happiness_correlation.png")


def plot_regional_averages(df: pd.DataFrame) -> None:
    """Horizontal bar chart of mean happiness score by region."""
    region_col = next(
        (c for c in df.columns if "region" in c.lower()), None
    )
    if region_col is None:
        print("  (Regional column not found — skipping regional chart)")
        return

    region_avg = (
        df.groupby(region_col)[TARGET]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    region_avg.plot(kind="barh", color="steelblue", ax=ax)
    ax.set_title("Average Happiness Score by Region", fontsize=13)
    ax.set_xlabel("Mean Happiness Score")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig("happiness_by_region.png", dpi=150)
    plt.close(fig)
    print("Saved: happiness_by_region.png")


def plot_gdp_vs_happiness(df: pd.DataFrame) -> None:
    """Scatter plot: GDP per capita vs. happiness score."""
    if "Logged GDP per capita" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.regplot(
        data=df,
        x="Logged GDP per capita",
        y=TARGET,
        scatter_kws={"alpha": 0.6, "s": 50},
        line_kws={"color": "firebrick"},
        ax=ax,
    )
    ax.set_title("GDP per Capita (log) vs. Happiness Score", fontsize=13)
    ax.set_xlabel("Logged GDP per capita")
    ax.set_ylabel("Happiness Score")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("happiness_vs_gdp.png", dpi=150)
    plt.close(fig)
    print("Saved: happiness_vs_gdp.png")


# ---------------------------------------------------------------------------
# 3. Regression model
# ---------------------------------------------------------------------------

def train_happiness_model(df: pd.DataFrame) -> None:
    """
    Multiple linear regression: predict happiness from socio-economic features.

    Steps:
        1. Select features and drop rows with NaN
        2. Scale features with StandardScaler
        3. Train/test split (80/20)
        4. Fit LinearRegression
        5. Report R² and RMSE; plot predicted vs. actual
    """
    available_features = [f for f in FEATURES if f in df.columns]
    clean = df[[TARGET] + available_features].dropna()

    X = clean[available_features].values
    y = clean[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n=== Multiple Linear Regression Results ===")
    print(f"  Features used : {available_features}")
    print(f"  Train size    : {len(y_train)}")
    print(f"  Test size     : {len(y_test)}")
    print(f"  R²            : {r2:.3f}")
    print(f"  RMSE          : {rmse:.3f}")

    # Feature importance (absolute standardised coefficients)
    coef_df = pd.DataFrame(
        {"Feature": available_features, "Coefficient": model.coef_}
    ).sort_values("Coefficient", key=abs, ascending=False)
    print("\n  Standardised coefficients (importance proxy):")
    print(coef_df.to_string(index=False))

    # Predicted vs. actual scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.7, edgecolors="white", linewidths=0.3, s=60)
    lims = [min(y_test.min(), y_pred.min()) - 0.2, max(y_test.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_title(f"Predicted vs. Actual Happiness (R² = {r2:.2f})", fontsize=13)
    ax.set_xlabel("Actual Happiness Score")
    ax.set_ylabel("Predicted Happiness Score")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("happiness_predicted_vs_actual.png", dpi=150)
    plt.close(fig)
    print("\nSaved: happiness_predicted_vs_actual.png")


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main() -> None:
    sns.set_theme(style="whitegrid", palette="tab10")

    df = load_data()

    print("--- Descriptive Statistics ---")
    print(df[[TARGET] + [f for f in FEATURES if f in df.columns]].describe().round(3))
    print()

    print("Generating charts …")
    plot_happiness_distribution(df)
    plot_correlation_heatmap(df)
    plot_regional_averages(df)
    plot_gdp_vs_happiness(df)

    train_happiness_model(df)

    print("\nAll done!  Check the PNG files in the current directory.")


if __name__ == "__main__":
    main()

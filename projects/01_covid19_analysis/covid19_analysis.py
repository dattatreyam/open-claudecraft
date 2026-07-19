"""
COVID-19 Global Trends Analysis
================================
Data source: Our World in Data COVID-19 public dataset
URL: https://github.com/owid/covid-19-data/tree/master/public/data
License: CC BY 4.0

Learning objectives:
    1. Download and explore a large real-world CSV with pandas
    2. Clean messy data (missing values, date parsing)
    3. Build rolling-average time-series charts
    4. Compare countries using bar plots and heat-maps
    5. Save publication-quality figures
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# 1. Data acquisition
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/"
    "public/data/owid-covid-data.csv"
)

COUNTRIES_OF_INTEREST = [
    "United States",
    "India",
    "Brazil",
    "United Kingdom",
    "Germany",
    "South Africa",
    "Japan",
]

# Columns we actually need (keeps memory usage low)
COLUMNS = [
    "iso_code",
    "location",
    "date",
    "new_cases_smoothed_per_million",
    "new_deaths_smoothed_per_million",
    "total_vaccinations_per_hundred",
    "people_fully_vaccinated_per_hundred",
    "total_deaths_per_million",
    "total_cases_per_million",
    "gdp_per_capita",
    "human_development_index",
]


def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Download the OWID COVID-19 dataset and return a cleaned DataFrame."""
    print(f"Downloading data from:\n  {url}\n")
    df = pd.read_csv(url, usecols=COLUMNS, parse_dates=["date"], low_memory=False)

    # Remove aggregate rows (continents / income groups / World)
    df = df[~df["iso_code"].str.startswith("OWID", na=False)].copy()

    # Sort for time-series operations
    df.sort_values(["location", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(
        f"Loaded {len(df):,} rows covering "
        f"{df['location'].nunique()} countries/territories "
        f"from {df['date'].min().date()} to {df['date'].max().date()}.\n"
    )
    return df


# ---------------------------------------------------------------------------
# 2. Analysis helpers
# ---------------------------------------------------------------------------

def daily_cases_chart(df: pd.DataFrame, countries: list[str]) -> None:
    """
    Line chart: smoothed new cases per million for selected countries.

    Technique: filter → pivot → plot.  The smoothed metric already applies
    a 7-day rolling average, so no further rolling is needed here.
    """
    subset = df[df["location"].isin(countries)].copy()
    subset = subset.dropna(subset=["new_cases_smoothed_per_million"])

    fig, ax = plt.subplots(figsize=(12, 6))
    for country, grp in subset.groupby("location"):
        ax.plot(grp["date"], grp["new_cases_smoothed_per_million"], label=country, lw=1.5)

    ax.set_title("COVID-19: Smoothed New Cases per Million (7-day avg)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("New cases per million")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig("covid19_daily_cases.png", dpi=150)
    plt.close(fig)
    print("Saved: covid19_daily_cases.png")


def cumulative_deaths_bar(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Horizontal bar chart: countries with the highest total deaths per million.

    Technique: grab the latest row per country, sort, and plot the top-N.
    """
    latest = (
        df.sort_values("date")
        .groupby("location", as_index=False)
        .last()
        .dropna(subset=["total_deaths_per_million"])
        .nlargest(top_n, "total_deaths_per_million")
        .sort_values("total_deaths_per_million")
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(latest["location"], latest["total_deaths_per_million"], color="steelblue")
    ax.set_title(f"Top {top_n} Countries by Total COVID-19 Deaths per Million", fontsize=14)
    ax.set_xlabel("Total deaths per million")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig("covid19_deaths_bar.png", dpi=150)
    plt.close(fig)
    print("Saved: covid19_deaths_bar.png")


def vaccination_over_time(df: pd.DataFrame, countries: list[str]) -> None:
    """
    Line chart: cumulative % of population fully vaccinated over time.

    Shows how vaccination campaigns unfolded at different speeds.
    """
    subset = df[df["location"].isin(countries)].copy()
    subset = subset.dropna(subset=["people_fully_vaccinated_per_hundred"])

    fig, ax = plt.subplots(figsize=(12, 6))
    for country, grp in subset.groupby("location"):
        ax.plot(grp["date"], grp["people_fully_vaccinated_per_hundred"], label=country, lw=1.5)

    ax.set_title("COVID-19: Share of Population Fully Vaccinated (%)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("People fully vaccinated (%)")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig("covid19_vaccination_timeline.png", dpi=150)
    plt.close(fig)
    print("Saved: covid19_vaccination_timeline.png")


def gdp_vs_vaccination_scatter(df: pd.DataFrame) -> None:
    """
    Scatter plot: GDP per capita vs. peak vaccination coverage.

    Demonstrates that wealthier countries generally achieved higher
    vaccination rates, though the relationship is imperfect.
    """
    latest = (
        df.sort_values("date")
        .groupby("location", as_index=False)
        .last()
        .dropna(subset=["gdp_per_capita", "people_fully_vaccinated_per_hundred"])
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        latest["gdp_per_capita"],
        latest["people_fully_vaccinated_per_hundred"],
        alpha=0.6,
        edgecolors="white",
        linewidths=0.3,
        s=50,
        color="steelblue",
    )

    # Trend line using numpy polyfit
    import numpy as np  # noqa: PLC0415

    x = latest["gdp_per_capita"].values
    y = latest["people_fully_vaccinated_per_hundred"].values
    coeffs = np.polyfit(x, y, 1)
    x_range = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_range, np.polyval(coeffs, x_range), color="firebrick", lw=2, label="Trend")

    ax.set_title("GDP per Capita vs. Vaccination Coverage", fontsize=14)
    ax.set_xlabel("GDP per capita (USD, log scale)")
    ax.set_ylabel("People fully vaccinated (%)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("covid19_gdp_vs_vaccination.png", dpi=150)
    plt.close(fig)
    print("Saved: covid19_gdp_vs_vaccination.png")


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise terminal summary of the global picture."""
    latest = (
        df.sort_values("date")
        .groupby("location", as_index=False)
        .last()
    )

    print("\n=== Global Summary (latest available data) ===")
    global_totals = df.groupby("date").agg(
        total_cases=("total_cases_per_million", "mean"),
        total_deaths=("total_deaths_per_million", "mean"),
    )
    last_row = global_totals.iloc[-1]
    print(f"  Average total cases per million  : {last_row['total_cases']:>10,.0f}")
    print(f"  Average total deaths per million : {last_row['total_deaths']:>10,.0f}")

    print("\n=== Top 5 Countries by Deaths per Million ===")
    top5 = (
        latest.dropna(subset=["total_deaths_per_million"])
        .nlargest(5, "total_deaths_per_million")[["location", "total_deaths_per_million"]]
    )
    print(top5.to_string(index=False))


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main() -> None:
    sns.set_theme(style="whitegrid", palette="tab10")

    df = load_data()

    print("Generating charts …")
    daily_cases_chart(df, COUNTRIES_OF_INTEREST)
    cumulative_deaths_bar(df)
    vaccination_over_time(df, COUNTRIES_OF_INTEREST)
    gdp_vs_vaccination_scatter(df)

    print_summary(df)

    print("\nAll done!  Check the PNG files in the current directory.")


if __name__ == "__main__":
    main()

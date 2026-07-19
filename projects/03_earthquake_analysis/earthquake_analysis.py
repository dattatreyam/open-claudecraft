"""
USGS Earthquake Catalog Analysis
==================================
Data source: USGS Earthquake Hazards Program – public Earthquake Catalog API
API docs:    https://earthquake.usgs.gov/fdsnws/event/1/
License:     Public domain (U.S. Government work)

Learning objectives:
    1. Query a public JSON/GeoJSON API with the `requests` library
    2. Parse GeoJSON features into a flat pandas DataFrame
    3. Explore the Gutenberg-Richter law (magnitude vs. frequency)
    4. Visualise earthquake locations on a world-map scatter plot
    5. Produce monthly time-series bar charts
"""

import sys
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fetch the most recent 365 days of M ≥ 4.5 earthquakes
QUERY_DAYS = 365
MIN_MAGNITUDE = 4.5

# USGS earthquake API endpoint
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Maximum events per API request (USGS cap = 20 000)
MAX_EVENTS = 20_000


# ---------------------------------------------------------------------------
# 1. Data acquisition
# ---------------------------------------------------------------------------

def fetch_earthquakes(days: int = QUERY_DAYS, min_mag: float = MIN_MAGNITUDE) -> pd.DataFrame:
    """
    Query the USGS Earthquake Catalog API and return a DataFrame.

    The API returns GeoJSON.  Each feature has:
        - geometry.coordinates : [longitude, latitude, depth_km]
        - properties.mag       : magnitude
        - properties.place     : human-readable location string
        - properties.time      : milliseconds since epoch (UTC)
        - properties.type      : event type (earthquake, quarry blast, …)
    """
    end_time = datetime.now(tz=timezone.utc)
    start_time = end_time - timedelta(days=days)

    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%d"),
        "endtime": end_time.strftime("%Y-%m-%d"),
        "minmagnitude": min_mag,
        "orderby": "time",
        "limit": MAX_EVENTS,
    }

    print(f"Querying USGS API for M≥{min_mag} events in the last {days} days …")
    try:
        resp = requests.get(USGS_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  API request failed: {exc}")
        print("  Using synthetic fallback data for demonstration.\n")
        return _synthetic_fallback()

    features = data.get("features", [])
    if not features:
        print("  No events returned by API — using synthetic fallback.\n")
        return _synthetic_fallback()

    records = []
    for feat in features:
        props = feat["properties"]
        lon, lat, depth = feat["geometry"]["coordinates"]
        records.append({
            "time": pd.to_datetime(props["time"], unit="ms", utc=True),
            "latitude": lat,
            "longitude": lon,
            "depth_km": depth,
            "magnitude": props.get("mag"),
            "place": props.get("place"),
            "event_type": props.get("type"),
        })

    df = pd.DataFrame(records)
    df = df[df["event_type"] == "earthquake"].copy()  # keep only tectonic earthquakes
    df.dropna(subset=["magnitude", "latitude", "longitude"], inplace=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  Loaded {len(df):,} earthquake events.\n")
    return df


def _synthetic_fallback() -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset so all plots still work
    when the API is unavailable (e.g., in a sandboxed environment).

    Magnitude distribution follows the Gutenberg-Richter law:
        log10(N) = a - b * M  →  sample via exponential distribution.
    Locations are biased toward known seismic belts (Ring of Fire).
    """
    rng = np.random.default_rng(42)
    n = 3000

    # Gutenberg-Richter: magnitude = M_min - log(U) / b
    b = 1.0
    magnitudes = MIN_MAGNITUDE - np.log(rng.uniform(size=n)) / b
    magnitudes = np.clip(magnitudes, MIN_MAGNITUDE, 9.5)

    # Ring of Fire: mix of Pacific-rim coordinates
    lat_centres = [35, -10, 0, -35, 55, -20, 15]
    lon_centres = [140, 120, -75, -70, 160, -175, 145]
    lats = np.concatenate([rng.normal(la, 10, n) for la in lat_centres])
    lons = np.concatenate([rng.normal(lo, 15, n) for lo in lon_centres])
    # Down-sample to exactly n points
    idx = rng.choice(len(lats), size=n, replace=False)
    lats = lats[idx]
    lons = lons[idx]

    end = datetime.now(tz=timezone.utc)
    times = [end - timedelta(seconds=int(rng.integers(0, QUERY_DAYS * 86400))) for _ in range(n)]

    df = pd.DataFrame({
        "time": pd.to_datetime(times, utc=True),
        "latitude": np.clip(lats, -90, 90),
        "longitude": ((lons + 180) % 360) - 180,
        "depth_km": rng.exponential(30, n),
        "magnitude": magnitudes,
        "place": ["Synthetic region"] * n,
        "event_type": ["earthquake"] * n,
    })
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Generated {len(df):,} synthetic earthquake events.\n")
    return df


# ---------------------------------------------------------------------------
# 2. Visualisations
# ---------------------------------------------------------------------------

def plot_world_map(df: pd.DataFrame) -> None:
    """
    Scatter plot on a world map: each dot is an earthquake coloured by magnitude.

    Technique: plot latitude/longitude as (y, x); use a diverging colour-map
    for magnitude.  No external basemap library is required — we use a simple
    filled rectangle for the oceans and a lighter background for land.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Simple ocean/land background
    ax.set_facecolor("#d0e8f5")
    ax.axhspan(-90, 90, color="#d0e8f5")

    sc = ax.scatter(
        df["longitude"],
        df["latitude"],
        c=df["magnitude"],
        cmap="YlOrRd",
        s=df["magnitude"] ** 2.5,   # size proportional to magnitude
        alpha=0.5,
        linewidths=0,
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Magnitude")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_title(
        f"Global Earthquakes M≥{MIN_MAGNITUDE} — last {QUERY_DAYS} days",
        fontsize=14,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig("earthquakes_world_map.png", dpi=150)
    plt.close(fig)
    print("Saved: earthquakes_world_map.png")


def plot_magnitude_distribution(df: pd.DataFrame) -> None:
    """
    Histogram (log y-axis) showing the Gutenberg-Richter law.

    The Gutenberg-Richter law states:
        log10(N) = a - b * M
    where N = cumulative count of earthquakes with magnitude ≥ M.
    The slope b ≈ 1 is universal across seismic regions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: histogram of magnitudes ---
    axes[0].hist(df["magnitude"], bins=30, color="steelblue", edgecolor="white", lw=0.4)
    axes[0].set_title("Magnitude Frequency Distribution")
    axes[0].set_xlabel("Magnitude")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # --- Right: Gutenberg-Richter cumulative plot ---
    mag_bins = np.arange(MIN_MAGNITUDE, df["magnitude"].max() + 0.1, 0.1)
    cumulative_n = [(df["magnitude"] >= m).sum() for m in mag_bins]

    axes[1].plot(mag_bins, cumulative_n, "o-", markersize=3, color="steelblue")
    axes[1].set_yscale("log")
    axes[1].set_title("Gutenberg-Richter Law (log scale)")
    axes[1].set_xlabel("Minimum Magnitude M")
    axes[1].set_ylabel("Cumulative count N (≥ M)")
    axes[1].grid(linestyle="--", alpha=0.4)

    # Fit a line to log10(N) vs. M and annotate b-value
    valid = np.array(cumulative_n) > 0
    if valid.sum() > 2:
        coeffs = np.polyfit(mag_bins[valid], np.log10(np.array(cumulative_n)[valid]), 1)
        b_value = -coeffs[0]
        axes[1].annotate(
            f"b ≈ {b_value:.2f}",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
            fontsize=11,
            color="firebrick",
        )

    fig.suptitle("Earthquake Magnitude Statistics", fontsize=14)
    fig.tight_layout()
    fig.savefig("earthquakes_magnitude.png", dpi=150)
    plt.close(fig)
    print("Saved: earthquakes_magnitude.png")


def plot_monthly_counts(df: pd.DataFrame) -> None:
    """
    Bar chart: number of earthquakes per month over the analysis period.

    Shows whether there are seasonal patterns (there generally aren't —
    earthquakes are approximately Poisson-distributed in time).
    """
    df = df.copy()
    df["month"] = df["time"].dt.tz_localize(None).dt.to_period("M")
    monthly = df.groupby("month").size().reset_index(name="count")
    monthly["month_str"] = monthly["month"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(monthly)), monthly["count"], color="steelblue", edgecolor="white", lw=0.4)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["month_str"], rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Monthly Earthquake Count (M≥{MIN_MAGNITUDE})", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of earthquakes")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig("earthquakes_monthly.png", dpi=150)
    plt.close(fig)
    print("Saved: earthquakes_monthly.png")


def plot_depth_vs_magnitude(df: pd.DataFrame) -> None:
    """
    Scatter plot: depth vs. magnitude, coloured by depth category.

    Categories: shallow (< 70 km), intermediate (70–300 km), deep (> 300 km).
    Deep earthquakes tend to have distinct magnitude distributions.
    """
    df = df.copy()
    depth_bins = [0, 70, 300, 700]
    depth_labels = ["Shallow (<70 km)", "Intermediate (70–300 km)", "Deep (>300 km)"]
    df["depth_category"] = pd.cut(df["depth_km"], bins=depth_bins, labels=depth_labels)

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {"Shallow (<70 km)": "steelblue", "Intermediate (70–300 km)": "orange", "Deep (>300 km)": "firebrick"}
    for cat, grp in df.groupby("depth_category", observed=True):
        ax.scatter(
            grp["depth_km"],
            grp["magnitude"],
            label=cat,
            alpha=0.4,
            s=20,
            color=palette.get(str(cat), "grey"),
        )
    ax.set_title("Earthquake Depth vs. Magnitude", fontsize=13)
    ax.set_xlabel("Depth (km)")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("earthquakes_depth_vs_magnitude.png", dpi=150)
    plt.close(fig)
    print("Saved: earthquakes_depth_vs_magnitude.png")


# ---------------------------------------------------------------------------
# 3. Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print key statistics to the terminal."""
    print("\n=== Earthquake Dataset Summary ===")
    print(f"  Total events          : {len(df):,}")
    print(f"  Date range            : {df['time'].min().date()} → {df['time'].max().date()}")
    print(f"  Magnitude range       : {df['magnitude'].min():.1f} – {df['magnitude'].max():.1f}")
    print(f"  Median magnitude      : {df['magnitude'].median():.1f}")
    print(f"  Median depth          : {df['depth_km'].median():.0f} km")

    print("\n  Magnitude counts:")
    for low in [4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
        high = low + 0.5
        n = ((df["magnitude"] >= low) & (df["magnitude"] < high)).sum()
        print(f"    M {low:.1f}–{high:.1f} : {n:>5,}")
    print(f"    M ≥7.0    : {(df['magnitude'] >= 7.0).sum():>5,}")


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main() -> None:
    sns.set_theme(style="whitegrid")

    df = fetch_earthquakes()

    print("Generating charts …")
    plot_world_map(df)
    plot_magnitude_distribution(df)
    plot_monthly_counts(df)
    plot_depth_vs_magnitude(df)

    print_summary(df)

    print("\nAll done!  Check the PNG files in the current directory.")


if __name__ == "__main__":
    main()

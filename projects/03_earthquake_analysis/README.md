# USGS Earthquake Catalog Analysis

**Data source:** [USGS Earthquake Hazards Program – Earthquake Catalog API](https://earthquake.usgs.gov/fdsnws/event/1/)  
**License:** Public domain (U.S. Government work)

## What you will learn

| Skill | Description |
|-------|-------------|
| REST API consumption | Query a public JSON/GeoJSON API with `requests` |
| JSON parsing | Convert nested GeoJSON into a flat pandas DataFrame |
| Statistical distributions | Gutenberg-Richter law (log-linear magnitude frequency) |
| Geo-visualisation | Scatter plot on a world map using latitude/longitude |
| Time-series analysis | Count earthquakes per month, detect seasonal patterns |

## Project structure

```
03_earthquake_analysis/
├── README.md
├── requirements.txt
└── earthquake_analysis.py   ← main analysis script
```

## Quick start

```bash
pip install -r requirements.txt
python earthquake_analysis.py
```

The script queries the USGS Earthquake Catalog API for all M ≥ 4.5 events
in a configurable date range and saves four PNG charts.

## Key questions explored

1. Where are earthquakes geographically concentrated?
2. How are magnitudes distributed? (hint: it follows the Gutenberg-Richter law)
3. What time of year / day do earthquakes occur? (spoiler: they are random)
4. Which regions experienced the most seismic activity?

## Concepts introduced

- `requests.get` + JSON parsing + GeoJSON feature extraction
- Log-scale histograms to visualise power-law distributions
- `matplotlib` scatter plot with colour-mapped magnitude
- Monthly time-series bar charts
- Defensive error handling for API calls

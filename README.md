# open-claudecraft
Upskilling with Claude AI: My Portfolio of Learning-Oriented Projects Using Public Data

---

## Projects

Each project folder contains a self-contained Python analysis script, a `README.md` with learning objectives, and a `requirements.txt`.

| # | Project | Public data source | Key skills |
|---|---------|-------------------|------------|
| 1 | [COVID-19 Global Trends Analysis](projects/01_covid19_analysis/) | [Our World in Data](https://ourworldindata.org/covid-cases) – CC BY 4.0 | pandas, time-series, matplotlib/seaborn |
| 2 | [World Happiness Report Analysis](projects/02_world_happiness/) | [World Happiness Report](https://worldhappiness.report/data/) – CC BY 4.0 | EDA, correlation, linear regression (scikit-learn) |
| 3 | [USGS Earthquake Catalog Analysis](projects/03_earthquake_analysis/) | [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/) – Public domain | REST API, GeoJSON, Gutenberg-Richter law |

---

## Quick start

```bash
# clone the repo, then run any project
cd projects/01_covid19_analysis
pip install -r requirements.txt
python covid19_analysis.py
```

---

## Requirements

- Python 3.10+
- See each project's `requirements.txt` for per-project dependencies.

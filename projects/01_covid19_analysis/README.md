# COVID-19 Global Trends Analysis

**Data source:** [Our World in Data – COVID-19 dataset](https://github.com/owid/covid-19-data/tree/master/public/data)  
**License:** CC BY 4.0

## What you will learn

| Skill | Description |
|-------|-------------|
| Data retrieval | Download a large public CSV directly from a URL with `pandas.read_csv` |
| Data cleaning | Handle missing values, parse dates, filter rows |
| Aggregation | Group-by operations, rolling averages |
| Visualisation | Line charts, bar charts, and a heat-map with `matplotlib` / `seaborn` |
| Storytelling | Turn numbers into a coherent narrative about the pandemic timeline |

## Project structure

```
01_covid19_analysis/
├── README.md
├── requirements.txt
└── covid19_analysis.py   ← main analysis script
```

## Quick start

```bash
pip install -r requirements.txt
python covid19_analysis.py
```

Running the script:
1. Downloads the latest Our World in Data COVID-19 CSV (≈ 100 MB).
2. Produces four PNG charts saved to the current directory.
3. Prints a short summary table to the terminal.

## Key questions explored

1. How did daily new cases evolve globally over time?
2. Which countries had the highest cumulative deaths per million?
3. How did vaccination roll-out vary between income groups?
4. Is there a correlation between GDP per capita and vaccination coverage?

## Concepts introduced

- `pd.read_csv` with a URL argument
- `pd.to_datetime` and `dt` accessor
- `.groupby()` + `.rolling()` for smoothed time-series
- `seaborn.heatmap` for correlation matrices
- Saving figures with `plt.savefig`

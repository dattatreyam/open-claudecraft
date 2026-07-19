# World Happiness Report Analysis

**Data source:** [World Happiness Report (WHR)](https://worldhappiness.report/data/)  
The dataset is embedded directly in the script using public values from the 2023 report,
and is also downloadable as a CSV from the official website.  
**License:** CC BY 4.0

## What you will learn

| Skill | Description |
|-------|-------------|
| Exploratory Data Analysis (EDA) | Distribution plots, descriptive statistics |
| Correlation analysis | Pearson correlation matrix and heat-maps |
| Regression | Simple & multiple linear regression with `scikit-learn` |
| Visualisation | Scatter plots, violin plots, choropleth-style bar charts |
| Feature interpretation | Understand which factors explain happiness most |

## Project structure

```
02_world_happiness/
├── README.md
├── requirements.txt
└── happiness_analysis.py   ← main analysis script
```

## Quick start

```bash
pip install -r requirements.txt
python happiness_analysis.py
```

## Key questions explored

1. What is the global distribution of happiness scores?
2. Which factors (GDP, social support, freedom, …) correlate most strongly with happiness?
3. How do regional averages differ?
4. Can we predict a country's happiness score from socio-economic indicators?

## Concepts introduced

- `pandas.DataFrame.describe()` for summary statistics
- `seaborn.heatmap` for a correlation matrix
- `seaborn.regplot` for scatter plots with regression lines
- `sklearn.linear_model.LinearRegression` for multi-variate regression
- `sklearn.metrics.r2_score` and `mean_squared_error` to evaluate the model

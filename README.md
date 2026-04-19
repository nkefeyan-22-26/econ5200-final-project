# ECON 5200: Consulting Report — Final Project
**Do Local Eviction Moratoriums Reduce Eviction Filing Rates?**
Northeastern University | Spring 2026

---

## Overview

This project estimates the causal effect of local eviction moratoriums on eviction filing rates during the COVID-19 pandemic using a Difference-in-Differences (DiD) identification strategy with Two-Way Fixed Effects (TWFE).

Data comes from the Princeton Eviction Lab's Eviction Tracking System (ETS), which provides weekly eviction filing counts for a panel of U.S. cities from 2020 onward. Filing rates are expressed as a ratio to each city's pre-pandemic baseline.

---

## Causal Question

Does having an active local eviction moratorium cause a reduction in eviction filing rates?

---

## Identification Strategy

**Method:** Staggered Difference-in-Differences (TWFE)

**Treated cities** (active local or state moratorium):
- Albuquerque, NM
- Boston, MA
- Bridgeport, CT
- Cincinnati, OH
- Cleveland, OH
- Columbus, OH

**Control cities** (federal CDC moratorium only):
- Charleston, SC
- Dallas, TX
- Fort Lauderdale, FL

**Key assumption:** Parallel trends — treated and control cities would have followed the same eviction filing trajectories absent the local moratorium policy.

---

## Data

- **Source:** Princeton Eviction Lab — Eviction Tracking System
- **URL:** https://evictionlab.org/eviction-tracking/get-the-data
- **Coverage:** 10 cities, weekly observations 2020–2024
- **N:** 2,644 city-week observations (after cleaning)
- **Outcome:** `filing_ratio` — weekly filings divided by pre-pandemic baseline
- **Treatment:** `moratorium` — 1 if an active local/state moratorium was in effect that week

---

## Preliminary Results

| Model | Estimate | 95% CI |
|---|---|---|
| Naive OLS | -0.553 | [-0.620, -0.486] |
| DiD (TWFE) | -0.040 | [-0.109, 0.030] |

The naive estimate is heavily upward-biased. After absorbing city and time fixed effects, the estimated effect of a local moratorium shrinks to -0.04 and is not statistically significant, suggesting local moratoriums may have added limited protection beyond the federal CDC moratorium already in effect.

---

## Repository Structure

```
econ5200-final-project/
├── notebook.ipynb                            # Main analysis notebook
├── eviction_tracking_all_sites_weekly.csv    # Raw data (Eviction Lab ETS)
├── app.py                                    # Streamlit dashboard
├── requirements.txt                          # Python dependencies
└── README.md
```

---

## How to Reproduce

```bash
git clone https://github.com/yourusername/econ5200-final-project
cd econ5200-final-project
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

---

## Deliverables

- [ ] Checkpoint notebook + proposal — due Apr 19
- [ ] Streamlit dashboard (deployed) — due Apr 26
- [ ] Executive Summary PDF — due Apr 26
- [ ] Technical Report PDF — due Apr 26
- [ ] Threats to Identification PDF — due Apr 26
- [ ] AI Methodology Appendix PDF — due Apr 26

---

## Tools & Libraries

- Python 3.11
- pandas, numpy, matplotlib, seaborn
- statsmodels (TWFE regression)
- streamlit, plotly (dashboard)

---

## AI Methodology

This project uses AI-augmented methodology documented via the P.R.I.M.E. framework (Prompt → Response → Iterate → Modify → Evaluate). All AI interactions are documented in the AI Methodology Appendix. All outputs were verified by the author.

---

*ECON 5200 — Causal Machine Learning & Applied Analytics — Northeastern University — Spring 2026*

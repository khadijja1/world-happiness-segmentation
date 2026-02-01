# World Happiness Report – K-Means Segmentation

> **Unsupervised ML pipeline** that clusters 156 countries into three distinct
> happiness tiers using K-Means, then surfaces the economic, social, and
> governance signals that define each tier.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Dataset at a Glance](#3-dataset-at-a-glance)
4. [Methodology](#4-methodology)
5. [Key Findings](#5-key-findings)
   - 5.1 [Correlation Landscape](#51-correlation-landscape)
   - 5.2 [Optimal Cluster Count – Elbow Analysis](#52-optimal-cluster-count--elbow-analysis)
   - 5.3 [Cluster Profiles & Centroids](#53-cluster-profiles--centroids)
   - 5.4 [Cluster Distribution](#54-cluster-distribution)
6. [Business Insights](#6-business-insights)
7. [Reproducing the Analysis](#7-reproducing-the-analysis)
   - 7.1 [Prerequisites](#71-prerequisites)
   - 7.2 [Installation](#72-installation)
   - 7.3 [Running the Notebook](#73-running-the-notebook)
8. [Tech Stack](#8-tech-stack)
9. [Project Roadmap](#9-project-roadmap)
10. [License](#10-license)

---

## 1. Project Overview

The **World Happiness Report** ranks every country on a composite *Happiness Score* built from six latent pillars. This project re-examines that data *without* using the Score itself as an input, instead letting an unsupervised K-Means model discover natural country groupings from the underlying six pillars alone.

The result is a clean three-cluster segmentation that aligns tightly with the original ranking — validating the report's methodology while also revealing *which* pillars differentiate clusters most sharply and *how* a nation can move between tiers.

### What makes this project stand out

| Aspect | Detail |
|---|---|
| **Unsupervised approach** | No target variable is fed to the model; clusters emerge purely from feature relationships |
| **End-to-end pipeline** | EDA → preprocessing → model selection → clustering → interpretation, all in a single notebook |
| **Dual visualisation layer** | Static Seaborn/Matplotlib plots for publication quality; interactive Plotly choropleth & scatter plots for exploration |
| **Inverse-transformed centroids** | Cluster centers are mapped back to original feature scales so findings are immediately interpretable |

---

## 2. Repository Layout

```
world-happiness-segmentation/
│
├── World_Happiness_Report_Segmentation.ipynb   ← End-to-end analysis notebook
├── world_happiness_data.csv                    ← Raw dataset (156 rows × 9 columns)
├── requirements.txt                            ← Pinned Python dependencies
├── README.md                                   ← This file
│
└── visualizations/                             ← Output directory for saved plots
    ├── pairplot.png
    ├── correlation_heatmap.png
    ├── elbow_method.png
    └── cluster_histograms/
```

---

## 3. Dataset at a Glance

| Attribute | Value |
|---|---|
| Source | World Happiness Report 2019 |
| Rows | 156 countries |
| Columns | 9 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Score range | 2.853 (South Sudan) – 7.769 (Finland) |
| Score mean ± std | 5.407 ± 1.113 |

### Feature Descriptions

| # | Feature | Description |
|---|---|---|
| 1 | `Overall rank` | Country's position in the happiness ranking (1–156) |
| 2 | `Country or region` | Nation name (categorical identifier) |
| 3 | `Score` | Composite happiness score (0–10 scale) |
| 4 | `GDP per capita` | Log GDP per capita, capturing economic prosperity |
| 5 | `Social support` | Proportion of people who say they have someone to count on |
| 6 | `Healthy life expectancy` | Average healthy years a newborn can expect to live |
| 7 | `Freedom to make life choices` | Population-level satisfaction with freedom |
| 8 | `Generosity` | Residual of regressing charitable donations on GDP |
| 9 | `Perceptions of corruption` | Average public perception of government & business corruption |

> **Clustering input:** columns 4–9 only. `Overall rank`, `Country or region`, and `Score` are deliberately excluded to keep the model fully unsupervised.

---

## 4. Methodology

The notebook follows a structured, reproducible pipeline. Each stage is clearly delineated with markdown headers inside the notebook.

```
┌─────────────────────┐
│  1. EDA             │  Summary stats · missing-value audit · duplicates
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  2. Visualisation   │  Pairplot · KDE distributions · correlation heatmap
│     (Static)        │  (Seaborn / Matplotlib)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  3. Visualisation   │  Scatter plots with trendlines · hover tooltips
│     (Interactive)   │  (Plotly Express / Graph Objects)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  4. Preprocessing   │  Drop target-leaking columns → StandardScaler
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  5. Model Selection │  Elbow Method (K = 1 … 19) → K = 3 chosen
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  6. K-Means         │  Fit · extract labels & centroids
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  7. Interpretation  │  Inverse-transform centroids · per-cluster histograms
│     & Visualisation │  Score vs Cluster · GDP vs Cluster · Choropleth map
└─────────────────────┘
```

### Preprocessing details

- **Dropped columns:** `Overall rank`, `Country or region`, `Score` — any column that either *is* the target or *directly encodes* the ranking is removed before clustering.
- **Scaling:** `StandardScaler` (zero mean, unit variance) applied to all six remaining features. This is mandatory for distance-based algorithms like K-Means, which are sensitive to feature magnitudes.
- **Final input shape:** 156 × 6

---

## 5. Key Findings

### 5.1 Correlation Landscape

The heatmap below (computed on all numeric columns except `Overall rank`) reveals which pillars move together most strongly.

| Feature Pair | Pearson r |
|---|---|
| GDP per capita ↔ Healthy life expectancy | **0.835** |
| Score ↔ GDP per capita | 0.794 |
| Score ↔ Healthy life expectancy | 0.780 |
| Score ↔ Social support | 0.777 |
| GDP per capita ↔ Social support | 0.755 |
| Score ↔ Freedom to make life choices | 0.567 |
| Score ↔ Perceptions of corruption | 0.386 |
| Score ↔ Generosity | **0.076** (weakest) |

**Takeaway:** GDP, life expectancy, and social support form a tightly correlated trio that dominates the Score. Generosity is nearly orthogonal to the composite score — it contributes meaningful signal to cluster separation but not to the overall ranking.

---

### 5.2 Optimal Cluster Count – Elbow Analysis

The inertia (within-cluster sum of squares) was plotted for K = 1 through 19. The sharpest rate-of-change drop — the "elbow" — occurs at **K = 3**, which was adopted as the final model.

---

### 5.3 Cluster Profiles & Centroids

After fitting, centroids were **inverse-transformed** back to original feature scales for direct interpretability.

#### Inverse-Transformed Cluster Centroids

| Feature | Cluster 0 — *Developing* | Cluster 1 — *Struggling* | Cluster 2 — *Prosperous* |
|---|---|---|---|
| GDP per capita | 1.047 | 0.450 | **1.345** |
| Social support | 1.332 | 0.875 | **1.471** |
| Healthy life expectancy | 0.825 | 0.447 | **0.954** |
| Freedom to make life choices | 0.385 | 0.327 | **0.543** |
| Generosity | 0.134 | 0.212 | **0.289** |
| Perceptions of corruption | 0.067 | 0.100 | **0.266** |

#### Scaled Centroids (model-space, for reference)

| Feature | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| GDP per capita | +0.356 | −1.146 | +1.108 |
| Social support | +0.413 | −1.118 | +0.880 |
| Healthy life expectancy | +0.412 | −1.152 | +0.949 |
| Freedom to make life choices | −0.055 | −0.460 | +1.053 |
| Generosity | −0.535 | +0.286 | +1.097 |
| Perceptions of corruption | −0.465 | −0.113 | +1.649 |

#### Cluster Narratives

**Cluster 2 – *Prosperous* (26 countries, 16.7%)**
The highest centroid on every single feature. These nations combine strong economies, robust social networks, long healthy lives, high personal freedom, generous charitable cultures, and — notably — the strongest corruption-perception scores. Examples from the top of the ranking: Finland, Denmark, Norway, Iceland, Netherlands.

**Cluster 0 – *Developing* (80 countries, 51.3%)**
The largest segment. GDP and social support sit comfortably above the global average, and life expectancy is moderate. Freedom and generosity scores are near the mean. Corruption perception is the lowest of all three clusters, suggesting either low actual corruption or low public awareness of it. This cluster represents the broad middle ground of global happiness.

**Cluster 1 – *Struggling* (50 countries, 32.1%)**
Centroids are below the global mean on GDP, social support, life expectancy, and freedom. Generosity is actually *higher* than Cluster 0, which is a recurring pattern in happiness research: nations with fewer material resources sometimes show stronger communal giving behaviour. Countries in this cluster occupy the bottom third of the happiness ranking.

---

### 5.4 Cluster Distribution

| Cluster | Label | Countries | Share |
|---|---|---|---|
| 0 | Developing | 80 | 51.3 % |
| 1 | Struggling | 50 | 32.1 % |
| 2 | Prosperous | 26 | 16.7 % |

The imbalance is itself a finding: prosperous nations are a minority globally, while the majority of countries cluster in the mid-tier developing segment.

---

## 6. Business Insights

| # | Insight | Implication |
|---|---|---|
| 1 | **GDP, life expectancy, and social support are nearly interchangeable predictors of happiness.** Correlation coefficients between all three and the Score exceed 0.77. | Investment in *any one* of these three pillars tends to co-move with the others. Public-health spending, for example, correlates strongly with both GDP growth and social cohesion. |
| 2 | **Generosity is the most independent pillar.** Its correlation with Score is only 0.076, yet it is the single feature with the largest centroid gap between Cluster 1 and Cluster 2 (0.212 → 0.289). | Charitable-giving programmes may be a low-cost lever for shifting a country's cluster membership without requiring macroeconomic change. |
| 3 | **Cluster 2 has the highest corruption-perception score (0.266 vs 0.067 for Cluster 0).** | Higher perceived corruption does not suppress happiness at the top tier — it likely reflects a more *transparent* and *informed* citizenry rather than more corruption. Governance transparency initiatives may reinforce, not undermine, happiness. |
| 4 | **51 % of countries fall in the mid-tier Cluster 0.** | The global "happiness gap" is concentrated between this large middle tier and the small prosperous tier (26 countries). Targeted improvements in freedom and generosity could be the fastest route for Cluster 0 nations to bridge that gap. |
| 5 | **The unsupervised clusters align with the supervised Score ranking.** Cluster 2 nations sit at the top of the ranking; Cluster 1 at the bottom. | The six pillars carry enough signal to reconstruct the ranking without ever seeing it — validating the World Happiness Report's composite methodology. |

---

## 7. Reproducing the Analysis

### 7.1 Prerequisites

| Software | Minimum Version |
|---|---|
| Python | 3.10+ |
| Jupyter Notebook / JupyterLab / VS Code (Jupyter extension) | Latest |

### 7.2 Installation

```bash
# 1. Clone the repository
git clone https://github.com/khadijja1/world-happiness-segmentation.git
cd world-happiness-segmentation

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 7.3 Running the Notebook

```bash
# Launch Jupyter from the project root
jupyter notebook World_Happiness_Report_Segmentation.ipynb
```

Then execute all cells top-to-bottom (**Kernel → Restart & Run All**).

> **Note:** The notebook reads the dataset via a relative path (`world_happiness_data.csv`). Keep the CSV in the same directory as the `.ipynb` file.

---

## 8. Tech Stack

| Layer | Library | Role |
|---|---|---|
| Data | `pandas` | DataFrame manipulation, I/O |
| Data | `numpy` | Vectorised numerical operations |
| ML | `scikit-learn` | `StandardScaler`, `KMeans` |
| Viz (static) | `matplotlib` | Elbow plot, histograms, layout |
| Viz (static) | `seaborn` | Pairplot, KDE distributions, heatmap |
| Viz (interactive) | `plotly` | Scatter plots with trendlines, choropleth map |
| Viz (interactive) | `chart-studio` | Plotly cloud hosting integration |
| Viz (interactive) | `bubbly` | Bubble-chart helpers |

---

## 9. Project Roadmap

Potential extensions that could build on this work:

- [ ] **Silhouette Score validation** — complement the Elbow Method with a silhouette analysis to quantify cluster quality.
- [ ] **PCA visualisation** — project the 6-D feature space down to 2 D for a scatter plot coloured by cluster label.
- [ ] **Temporal analysis** — incorporate multiple years of Happiness Report data to track whether countries migrate between clusters over time.
- [ ] **Supervised benchmark** — train a simple classifier (e.g., Random Forest) on the cluster labels to rank feature importance and compare with the unsupervised centroid deltas.
- [ ] **Dashboard** — wrap the choropleth map and cluster summaries into a Dash or Streamlit interactive app.

---

## 10. License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 Khadija Faisal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
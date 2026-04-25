# Customer Segmentation + Pricing Strategy

> End-to-end ML pipeline on **1M+ real retail transactions**.  
> KMeans · DBSCAN · GMM · Gradient Boosting WTP model · SHAP explainability · Pricing recommendations.

---

## Results

| Metric | Value |
|---|---|
| Dataset | UCI Online Retail II — 1,067,371 real transactions |
| Date range | Dec 2009 – Dec 2011 |
| Customers after cleaning | ~4,300 unique |
| Segments identified | 4 (GMM, best BIC) |
| GMM silhouette score | 0.68 |
| WTP model R² | 0.79 |
| High-risk churn flagged | ~17% of customer base |

---

## Project Structure

```
retail-segmentation/
│
├── data/
│   ├── load_data.py        Download UCI Online Retail II via ucimlrepo
│   └── clean_data.py       Remove nulls, cancellations, bad codes, dupes
│
├── src/
│   ├── features.py         RFM scoring + 8 behavioral signals per customer
│   ├── clustering.py       KMeans / DBSCAN / GMM with full metric comparison
│   ├── churn.py            Weighted churn risk score [0,1] from RFM signals
│   ├── wtp_model.py        GBR willingness-to-pay model + SHAP explainability
│   ├── pricing.py          Tier recommendations + MRR uplift projections
│   └── visualize.py        9 publication-quality plots saved to outputs/
│
├── tests/
│   ├── test_clean_data.py  8 tests — cleaning pipeline
│   ├── test_features.py    11 tests — RFM feature engineering
│   ├── test_clustering.py  13 tests — all three models
│   ├── test_churn.py       9 tests — churn risk scoring
│   └── test_wtp_model.py   12 tests — WTP model + SHAP
│
├── notebooks/
│   └── full_analysis.ipynb Cell-by-cell walkthrough
│
├── outputs/                Generated plots (gitignored)
├── main.py                 Single entry point — runs everything
├── Makefile                Common commands
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/retail-segmentation.git
cd retail-segmentation

# 2. Virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Run — downloads UCI data automatically on first run
python main.py
```

After the first run, skip the download:

```bash
python main.py --skip-download
python main.py --skip-download --no-plots   # fastest
```

---

## Methodology

### Data — UCI Online Retail II

Real transactional data from a UK-based gift retailer (2009–2011).  
1,067,371 raw rows → ~750,000 after cleaning → ~4,300 unique customers.

**Cleaning steps:**
- Drop rows with missing `CustomerID` (can't segment without an ID)
- Drop cancelled invoices (`InvoiceNo` starts with `C`)
- Drop zero/negative `Quantity` and `UnitPrice`
- Drop non-product stock codes (`POST`, `BANK CHARGES`, etc.)
- Drop exact duplicate rows

### Feature Engineering

One row per customer. Features built from transaction history:

| Feature | Description |
|---|---|
| `recency_days` | Days since last purchase |
| `frequency` | Number of unique orders |
| `monetary` | Total spend (£) |
| `avg_order_value` | monetary / frequency |
| `avg_items_per_order` | Total items / orders |
| `unique_products` | Distinct stock codes purchased |
| `active_days` | Days between first and last purchase |
| `revenue_per_day` | monetary / active_days |

RFM quintile scores (1–5) derived from the above.

### Clustering — Model Selection

Three algorithms compared on the same feature matrix:

| Model | Silhouette | Notes |
|---|---|---|
| KMeans (k=4) | 0.61 | Fast baseline; sensitive to outliers |
| DBSCAN (ε=0.5) | 0.54 | Finds outliers; over-segments |
| **GMM (k=4)** | **0.68** | **Selected** — best BIC, soft probabilities |

**Why GMM?**  
Soft probabilities let each customer carry uncertainty (e.g. 70% Champion, 30% At-Risk). These probabilities become features in the WTP model.

### Segments

| Segment | Description |
|---|---|
| Champions | High RFM across all three dimensions. Recently active, frequent, high spend. |
| Loyal Customers | Strong frequency and monetary, slightly older recency. |
| At Risk | Previously good customers showing declining recency signal. |
| Needs Attention | Low RFM. Infrequent, low-spend, long since last purchase. |

### Churn Risk

Weighted composite of inverse-ranked RFM dimensions:

```
churn_risk = recency_rank×0.5 + (1-frequency_rank)×0.3 + (1-monetary_rank)×0.2
```

Purely unsupervised — no churn labels needed.

### WTP Model

Since the dataset has no explicit WTP labels, a **behavioural proxy** is engineered:

```
wtp_proxy = avg_order_value × log1p(frequency) × recency_weight
recency_weight = 1 / (1 + recency_days/30)
```

A **Gradient Boosting Regressor** (400 trees, lr=0.05, depth=4) predicts this proxy from RFM + behavioral + GMM soft-probability features.  

**SHAP explainability** reveals which features drive predictions — top driver is `avg_order_value` (SHAP=0.34), followed by `frequency` and GMM membership probabilities.

### Pricing Strategy

Four tiers anchored to segment WTP distributions:

| Tier | Price | Target Segment |
|---|---|---|
| Enterprise | £349/mo | Champions |
| Pro | £79/mo | Loyal Customers |
| Win-Back | £47/mo | At Risk (40% off, 3 months) |
| Starter | £0 | Needs Attention (freemium) |

---

## Plots Generated

| File | Description |
|---|---|
| `01_rfm_scatter.png` | Customers in RFM space by segment |
| `02_elbow_curve.png` | KMeans inertia + silhouette vs k |
| `03_gmm_bic.png` | GMM BIC/AIC vs n_components |
| `04_gmm_clusters_pca.png` | 2D PCA projection of GMM clusters |
| `04_kmeans_clusters_pca.png` | 2D PCA projection of KMeans clusters |
| `05_churn_distribution.png` | Churn risk histograms + boxplots |
| `06_segment_revenue.png` | Total revenue by segment |
| `07_wtp_distribution.png` | WTP score distribution per segment |
| `08_shap_summary.png` | SHAP beeswarm — WTP feature importance |
| `09_pricing_strategy.png` | MRR uplift per tier |

---

## Tests

```bash
pytest tests/ -v           # all 53 tests
pytest tests/ -q           # quiet summary
pytest tests/ --cov=src    # with coverage
```

---

## Dataset Citation

Chen, D. (2012). *Online Retail II* [Dataset].  
UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D  
License: CC BY 4.0

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Core language |
| scikit-learn | 1.4+ | KMeans, DBSCAN, GMM, GBR |
| SHAP | 0.44+ | Model explainability |
| pandas | 2.1+ | Data manipulation |
| numpy | 1.26+ | Numerical computing |
| matplotlib | 3.8+ | Visualisations |
| ucimlrepo | 0.0.3+ | Dataset download |
| pytest | 8.0+ | Unit testing |

---

## License

MIT — free to use, modify, and distribute.

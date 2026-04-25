# Methodology

Deep-dive on every technical decision in the pipeline.
Use this when someone asks "why did you choose X?" in an interview.

---

## 1. Dataset choice — UCI Online Retail II

**Why this dataset over alternatives:**

| Alternative | Problem |
|---|---|
| Mall Customer Segmentation | 200 rows — meaningless clustering |
| IBM Telco Churn | No transaction history — can't do RFM |
| Instacart Orders | No revenue column — can't do monetary |
| Synthetic data | Not real — interviewers discount it |

UCI Online Retail II gives 1M+ real transactions with a full time
dimension (InvoiceDate) — so Recency is genuine, not engineered.
It's cited in 100+ published papers, so interviewers recognise it.

---

## 2. Cleaning decisions

**Why drop missing CustomerID?**  
RFM requires grouping by customer. A transaction with no customer ID
can't contribute to any segment. Imputation is impossible — you can't
infer who made a purchase.

**Why drop cancellations (C prefix)?**  
Cancellations represent returned goods. Including them would inflate
frequency (the customer didn't actually complete an order) and deflate
monetary (the refund reduces revenue). Both distort the segments.

**Why drop zero/negative Quantity and UnitPrice?**  
Zero quantity = a test entry or line with no product.
Negative quantity = a return (not covered by the 'C' filter for all systems).
Zero price = free samples / gifts — not real commercial transactions.

**Why drop bad stock codes (POST, BANK CHARGES, etc.)?**  
These are internal operational codes, not products. Including them
inflates frequency (a customer is counted as buying "postage") and
pollutes the product affinity signals.

---

## 3. Feature engineering

**Why RFM specifically?**  
RFM is the industry standard for customer segmentation because:
- It is interpretable (you can explain every score to a non-technical stakeholder)
- It captures the three axes that predict CLV: when, how often, how much
- It is entirely derivable from a transaction log — no surveys needed

**Why quintile scoring (1–5) instead of raw values?**  
Raw values are heavily skewed (a few customers spend 100× the median).
Quintiles are rank-based and outlier-robust. A score of 5 means "top 20%",
regardless of whether the top buyer spent £500 or £50,000.

**Why `rank(method='first')` for frequency?**  
Frequency is discrete with many ties (multiple customers with exactly 3 orders).
`rank(method='first')` breaks ties by order of appearance, ensuring a
clean 5-bucket split rather than an uneven one.

**Why add `avg_order_value`, `unique_products`, `active_days`?**  
Pure RFM misses important variation:
- Two customers with identical RFM can have wildly different order sizes
- Product breadth predicts cross-sell opportunity
- Active days distinguishes a customer who bought 10× in a week
  from one who bought 10× over two years

---

## 4. Clustering model selection

**Why three models?**

Showing multiple approaches and then justifying the selection
demonstrates proper ML thinking. You're not just using sklearn defaults —
you're comparing, measuring, and deciding.

**Why KMeans is the baseline:**  
Fast, interpretable, widely understood. Sets the benchmark.
Weakness: assumes spherical clusters and is sensitive to outliers.

**Why DBSCAN matters even if not selected:**  
It identifies genuine outliers (noise points = customers who don't
fit any segment). On this dataset, ~8% are noise — these can be treated
as a "micro-segment" of one-off buyers. Also shows you understand
density-based approaches.

**Why GMM is selected:**

1. **Soft probabilities**: A hard label loses information. A customer
   at the Champion/Loyal boundary gets a single label, discarding the
   fact that they are 55% Champion and 45% Loyal. GMM preserves this.
   The probabilities become features in the WTP model.

2. **Elliptical clusters**: RFM data is skewed — log-normally distributed
   in all three dimensions. KMeans' spherical assumption is wrong.
   GMM's full covariance matrix fits elliptical shapes.

3. **Principled model selection**: BIC (Bayesian Information Criterion)
   penalises model complexity. Minimising BIC finds the n_components that
   best explains the data without overfitting. No arbitrary choice.

**How to pick k:**  
- Elbow curve: look for the k where inertia stops dropping sharply
- Silhouette score: pick the k with the highest average score
- GMM BIC: pick the n_components with the lowest BIC
All three converge on k=4 for this dataset.

---

## 5. Churn risk scoring

**Why not train a classification model?**  
The UCI dataset has no churn labels — a customer's last purchase in
December 2011 doesn't tell us if they churned (the dataset ends then).
Supervised churn requires knowing who *actually* churned.

**The unsupervised approach:**  
We define churn risk as a function of behavioural signals that correlate
with churn in the literature:
- Recency is the strongest predictor (50% weight) — if they haven't
  bought in a long time, they're at risk
- Frequency (30%) — habitual buyers are less likely to churn
- Monetary (20%) — high spenders have more commitment to the brand

Percentile ranking makes this robust to outliers. The formula is
transparent and explainable.

---

## 6. WTP model design

**Why a proxy target instead of real WTP labels?**  
Real WTP requires a survey (Van Westendorp PSM, Gabor-Granger, or
conjoint analysis). We don't have that for the UCI dataset.
The proxy captures the same logic:
- Customers who spend more per order have demonstrated willingness to pay
- Customers who buy frequently have embedded the habit — harder to lose
- Customers who bought recently are "warmer" and more likely to respond to pricing

**Why Gradient Boosting over linear regression?**  
- GBR handles non-linear interactions (high frequency + high AOV compounds)
- Robust to the skewed distributions in RFM data
- `n_iter_no_change=20` gives automatic early stopping — avoids overfitting
- Works naturally with SHAP explainability

**Why SHAP?**  
- Provides per-customer, per-feature attributions
- Satisfies the "which features matter?" question for any stakeholder
- Passes the test of reproducibility — same model = same SHAP values
- Interview answer: "The top driver of WTP is avg_order_value
  (SHAP=0.34), meaning customers who spend more per visit predict
  higher willingness to pay — as expected."

---

## 7. Pricing strategy

**Why four tiers?**  
Four tiers match the four segments precisely. This is the key insight:
segmentation is not an end in itself — it exists to enable differentiated
pricing and targeting. The segments are designed to have meaningfully
different WTP distributions.

**Why a freemium entry tier?**  
Needs Attention customers have low WTP. Charging them upfront risks
churn before they convert. Freemium gives them a low-barrier entry,
usage caps create natural upgrade triggers, and the conversion happens
after demonstrated value.

**Why 40% off for At Risk?**  
At Risk customers have *already demonstrated value* — they are not
low-value customers, they are dormant high-value ones. A reactivation
discount is more efficient than acquiring new customers.
The time limit (3 months) prevents them staying on the discount
indefinitely.

---

## 8. Testing strategy

**Unit tests, not integration tests:**  
The pipeline depends on a 1M-row download. Integration tests would
be slow and brittle. Unit tests use small synthetic fixtures that
exercise the logic in isolation.

**What is tested:**
- Data cleaning: each cleaning step removes exactly what it should
- Feature engineering: RFM scores have correct ranges and direction
- Clustering: all models return expected shapes and metric ranges
- Churn risk: monotonicity (higher recency = higher risk)
- WTP model: R² > 0, non-negative predictions, SHAP covers all features

**What is not tested:**  
End-to-end pipeline (covered by `main.py` run manually) and
visualisation output (visual inspection is sufficient).

# Interview Q&A

Answers specific to this project. Practice saying these out loud.

---

## "Walk me through your project."

"I built an end-to-end customer segmentation pipeline on a real dataset —
1 million transactions from a UK online retailer. I cleaned the raw data,
engineered RFM features, then compared three clustering algorithms:
KMeans as a baseline, DBSCAN for outlier detection, and GMM which I
selected because it gives soft probability scores that feed directly
into a downstream WTP model. I then built a Gradient Boosting model
to predict willingness-to-pay from behavioural signals, used SHAP to
explain which features drove the predictions, and translated the segment
profiles into a four-tier pricing strategy with estimated MRR uplift.
The whole thing is tested with 50+ unit tests and documented in a Jupyter notebook."

---

## "Why did you use GMM instead of KMeans?"

"Three reasons. First, GMM produces soft probabilities — a customer can
be 70% Champion and 30% Loyal, which is more realistic than a hard
assignment. Those probabilities become features in the WTP model.
Second, GMM uses a full covariance matrix which fits the elliptical
cluster shapes in RFM data better than KMeans' spherical assumption.
Third, BIC gives a principled way to select the number of components —
I don't have to eyeball an elbow curve."

---

## "How did you pick k=4?"

"I used two methods in parallel. For KMeans, I plotted inertia vs k
(the elbow curve) and the silhouette score vs k — both showed a clear
knee at k=4. For GMM, I ran a BIC search across n_components 2 to 6
and the minimum BIC was at 4. Having two independent methods converge
on the same answer gives me confidence."

---

## "What is SHAP and why did you use it?"

"SHAP stands for SHapley Additive exPlanations. It comes from cooperative
game theory and assigns each feature a contribution to a specific
prediction. For my WTP model, it told me that avg_order_value was the
top driver (SHAP=0.34), meaning customers who spend more per order are
predicted to have higher willingness-to-pay. I used it because it gives
per-customer, per-feature explanations — not just a global feature
importance — and because it passes the consistency test: if a feature's
effect increases, its SHAP value increases."

---

## "How did you handle the fact that you had no WTP labels?"

"The UCI dataset is a transaction log — there are no survey-based WTP
measurements. So I engineered a proxy target from behavioural signals.
The formula is: WTP proxy = avg_order_value × log1p(frequency) × recency_weight,
where recency_weight decays exponentially with days since last purchase.
This captures the three dimensions of WTP: willingness to pay per transaction,
habit strength, and current engagement. It's a proxy, not ground truth —
in a real project you'd run a Van Westendorp PSM survey. I document this
assumption clearly."

---

## "How did you validate your clusters?"

"I used three metrics. Silhouette score measures how similar each point
is to its own cluster vs other clusters — GMM scored 0.68, which is
strong. Davies-Bouldin index measures cluster compactness and separation —
lower is better. And for GMM specifically, BIC gives a model-level
measure of fit penalised for complexity. I also did a sanity check:
Champions had the highest average spend, lowest recency, and highest
frequency — exactly what you'd expect from the definition."

---

## "What would you do differently with more time?"

"Three things. First, I'd run a Van Westendorp PSM survey on a sample
of customers in each segment to get real WTP labels instead of the
proxy. Second, I'd add a cohort analysis — looking at how segment
membership changes over time for individual customers, to measure
segment stability. Third, I'd productionise the pipeline with
Airflow or Prefect so it runs on a schedule as new transactions come in,
rather than being a one-shot analysis."

---

## "How did you handle the skewed distributions in RFM?"

"Two ways. For the clustering features, I used StandardScaler — this
centres and scales each feature, which reduces the dominance of
monetary (which has a much wider range than recency or frequency).
For the RFM scores, I used quintile ranking, which is rank-based and
completely robust to outliers. A customer who spent £50,000 and one
who spent £5,000 both score 5 if they're in the top 20% — the raw
values don't matter."

---

## "How many tests do you have and what do they cover?"

"53 unit tests across five test files. I test the cleaning pipeline
(each cleaning step removes exactly the right rows), feature engineering
(RFM scores are in range, frequency counts unique invoices correctly),
all three clustering models (output shapes, silhouette ranges,
probability sums), churn risk (monotonicity — higher recency means
higher churn risk), and the WTP model (positive R², non-negative
predictions, SHAP covers all features). I don't test end-to-end
because that requires downloading 1M rows — that's manual validation."

---

## "What did you find most interesting in the data?"

"The At Risk segment was the most interesting. These are customers with
good historical RFM — they've spent a meaningful amount and ordered
multiple times — but their recency signal has gone cold. They've lapsed.
That's actually a better opportunity than the Needs Attention segment,
because they've already demonstrated value. A time-limited win-back
offer is more efficient than acquiring a new customer. The data makes
that case clearly."

# 🌱 Climate Resilience and Agricultural Technology Adoption
### Smallholder Farmers in Kenya – Analysis & Decision-Support Tool

> **Developed as part of an application for the Global R&D Data Analyst role at One Acre Fund**

---

##  Project Overview

This project investigates the factors influencing adoption of modern agricultural practices among smallholder farmers in Kenya, and evaluates how climate risks, financial inclusion, and farmer characteristics affect agricultural resilience.

The work is structured to meet the analytical standards required for agricultural development programmes — combining causal inference framing, mixed-effects modelling, machine learning, and an interactive deployment tool.

---

##  Interactive Dashboard (Tableau)

Explore the visual story of this analysis on Tableau Public:

🔗 **[View the Tableau Dashboard](https://public.tableau.com/app/profile/sarah.owendi/viz/Agriculture_17805196846110/Story1)**

The dashboard covers:
- Adoption rates by region, education, and financing access
- Climate hazard prevalence and vulnerability mapping
- Farmer demographic profiles

---

##  Project Structure

```
├── Agri_Strengthened.ipynb     # Main analysis notebook
├── app.py                      # Streamlit deployment app
├── Agri.csv                    # Raw survey data (not included in repo)
├── cleaned_agriculture_data.csv# Output of cleaning pipeline
└── README.md                   # This file
```

---

##  Research Questions

1. Does access to financing increase adoption of modern farming practices?
2. Which farmer groups are most vulnerable to climate-related losses?
3. What factors predict adoption of improved agricultural practices?
4. How can data-driven recommendations improve agricultural resilience?

---

##  Notebook Summary (`Agri_Strengthened.ipynb`)

| Section | Description |
|---|---|
| 1. Imports | All libraries with reproducibility seed set |
| 2. Load Data | Shape audit, missingness report |
| 3. Data Cleaning | Mode/median imputation, regex extraction for numeric fields |
| 4. Feature Engineering | Adoption score (0–3), high adoption binary, climate risk score (0–5) |
| 5. EDA | Adoption by education, gender, financing, region; climate hazard prevalence |
| 6. Statistical Analysis | Chi-square + Cramér's V; mixed-effects logistic regression (regional random intercept); confounder analysis; RCT design proposal with power calculation |
| 7. Climate Vulnerability | High-risk farmer profiling; Mann-Whitney U test; hazard prevalence chart |
| 8. Machine Learning | Logistic Regression, Random Forest, Gradient Boosting; 5-fold CV; ROC-AUC; feature importance |
| 9. Sensitivity Analysis | Adoption threshold robustness check (1, 2, or 3 practices) |
| 10. Save Output | Cleaned dataset export |
| 11. Findings | Summary table, programme recommendations, methodological limitations |

---

##  Deployment App (`app.py`)

A Streamlit web app with three tabs:

### Tab 1 – Single Farmer Prediction
Input a farmer's profile and get:
- Adoption probability score
- Predicted class (High / Low Adopter)
- Climate vulnerability rating
- Recommended programme action

### Tab 2 – Batch Scoring
Upload a CSV of farmers → download a scored file with priority tiers (High / Medium / Standard).

### Tab 3 – Programme Dashboard
Upload the cleaned dataset for regional and demographic analytics charts.

### Running the app

```bash
# Install dependencies
pip install streamlit pandas numpy matplotlib seaborn scikit-learn statsmodels

# Run
streamlit run app.py
```

To use your trained model in the app, save it after running the notebook:

```python
import joblib
joblib.dump(best_pipe, 'model_pipeline.pkl')
```

Then place `model_pipeline.pkl` in the same folder as `app.py`.

---

##  Dependencies

```
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
streamlit
joblib
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn streamlit joblib
```

---

##  Key Findings

| Finding | Method | Confidence |
|---|---|---|
| Financing access significantly associated with adoption | Chi-square + Cramér's V | High |
| Regional clustering explains meaningful adoption variance | Mixed-effects GLMM | Moderate |
| Internet access is a strong adoption predictor | Random Forest feature importance | Moderate |
| Education level positively correlated with adoption | EDA + ordinal regression | High |
| High climate risk farmers show different adoption patterns | Mann-Whitney U | Moderate |

---

##  Methodological Notes

- This is **observational, cross-sectional data** — associations identified are correlational, not causal
- A **cluster-RCT design** is proposed in Section 6.3 to establish causal impact of financing on adoption
- Self-reported climate loss data may carry recall bias
- The adoption index covers 3 practices only; other innovations are not captured

---

##  Author

**Sarah Owendi**
Nairobi, Kenya
[Tableau Profile](https://public.tableau.com/app/profile/sarah.owendi)

"""
Agricultural Technology Adoption Predictor
One Acre Fund – Farmer Targeting Tool
Deployed via Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agri Adoption Predictor",
    page_icon="🌱",
    layout="wide"
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FEATURES = [
    'education level', 'gender', 'age group',
    'agricultural financing', 'phone ownership', 'internet use',
    'farming experience', 'climate_risk_score'
]
TARGET = 'high_adoption'

CAT_COLS = ['education level', 'gender', 'age group',
            'agricultural financing', 'phone ownership', 'internet use']
NUM_COLS = ['farming experience', 'climate_risk_score']

# ── Model training ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train and return the best pipeline, fitted on the full dataset."""
    available = [f for f in FEATURES if f in df.columns]
    cat = [c for c in CAT_COLS if c in available]
    num = [c for c in NUM_COLS if c in available]

    X = df[available].copy()
    y = df[TARGET].copy()

    preprocessor = ColumnTransformer([
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat),
        ('num', StandardScaler(), num)
    ], remainder='drop')

    models = {
        'Logistic Regression': Pipeline([
            ('pre', preprocessor),
            ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        'Random Forest': Pipeline([
            ('pre', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=6,
                                            min_samples_leaf=5, random_state=RANDOM_STATE))
        ]),
        'Gradient Boosting': Pipeline([
            ('pre', preprocessor),
            ('clf', GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                learning_rate=0.05, random_state=RANDOM_STATE))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = {}
    for name, pipe in models.items():
        s = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        scores[name] = s.mean()

    best_name = max(scores, key=scores.get)
    best_pipe = models[best_name]
    best_pipe.fit(X, y)

    return best_pipe, best_name, scores, available, cat, num


@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)


def make_synthetic_data(n=400):
    np.random.seed(42)
    edu = np.random.choice(['no formal education','primary','secondary','tertiary'],
                           n, p=[0.15,0.40,0.30,0.15])
    fin = np.random.choice(['yes','no'], n, p=[0.35,0.65])
    net = np.random.choice(['yes','no'], n, p=[0.45,0.55])
    edu_score = pd.Series(edu).map({'no formal education':0,'primary':1,'secondary':2,'tertiary':3})
    fin_score = (pd.Series(fin) == 'yes').astype(int)
    net_score = (pd.Series(net) == 'yes').astype(int)
    adopt_score = (edu_score * 2 + fin_score * 2 + net_score +
                   np.random.randint(0, 2, n))
    high_adopt = (adopt_score >= 4).astype(int)
    risk_cols = {
        'losses-rain pattern': np.random.choice([0,1], n, p=[0.5,0.5]),
        'losses-drought':      np.random.choice([0,1], n, p=[0.6,0.4]),
        'losses-heatwave':     np.random.choice([0,1], n, p=[0.7,0.3]),
        'losses-storms':       np.random.choice([0,1], n, p=[0.75,0.25]),
        'losses-mudslides':    np.random.choice([0,1], n, p=[0.8,0.2]),
    }
    FEATURES = [
    'education level', 'gender', 'age group',
    'agricultural financing', 'phone ownership', 'internet use',
    'farming experience', 'climate_risk_score'
]
TARGET = 'high_adoption'
HIGH_ADOPTION_COL = 'high_adoption'

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/One_Acre_Fund_logo.svg/320px-One_Acre_Fund_logo.svg.png",
             use_container_width=True)
    st.markdown("---")
    st.header("📂 Upload Your Data")
    uploaded = st.file_uploader(
        "Upload `cleaned_agriculture_data.csv`",
        type=["csv"],
        help="Output from Agri_Strengthened.ipynb"
    )
    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown("""
    **Model predicts:** High adoption of modern agricultural practices
    (fertiliser + certified seeds + pest management)

    **Algorithm:** Best of Logistic Regression, Random Forest, Gradient Boosting
    (chosen by 5-fold cross-validated ROC-AUC)

    **Data source:** Kenya smallholder farmer survey

    🔗 [Tableau Dashboard](https://public.tableau.com/app/profile/sarah.owendi/viz/Agriculture_17805196846110/Story1)
    """)
    st.markdown("---")
    st.caption("Built by Sarah Owendi · Nairobi, Kenya")

# ── Load data & train 
if uploaded:
    df = load_data(uploaded)
    data_source = "uploaded"
else:
    df = make_synthetic_data()
    data_source = "demo"

if TARGET not in df.columns:
    st.error(f"Column `{TARGET}` not found. Please upload `cleaned_agriculture_data.csv` from your notebook.")
    st.stop()

with st.spinner("Training model..."):
    pipe, best_name, cv_scores, avail_features, cat_cols, num_cols = train_model(df)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌱 Agricultural Technology Adoption Predictor")
st.markdown("> Predicts likelihood of smallholder farmers adopting modern practices — for extension service targeting and financing prioritisation.")

if data_source == "demo":
    st.info("⚡ **Demo mode** – running on synthetic data. Upload `cleaned_agriculture_data.csv` in the sidebar to use your real data.", icon="ℹ️")
else:
    st.success(f"✅ Data loaded: **{len(df):,} farmers** · Model: **{best_name}** · CV AUC: **{max(cv_scores.values()):.3f}**")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Prediction",
    "📋 Batch Scoring",
    "📊 Dashboard",
    "🤖 Model Performance"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Predict for one farmer")

    c1, c2 = st.columns(2)
    with c1:
        education  = st.selectbox("Education Level",
                                   ['no formal education','primary','secondary','tertiary'])
        gender     = st.selectbox("Gender", ['male','female'])
        age_group  = st.selectbox("Age Group", ['18-30','31-45','46-60','60+'])
        financing  = st.selectbox("Agricultural Financing Access", ['yes','no'])
    with c2:
        phone      = st.selectbox("Phone Ownership", ['yes','no'])
        internet   = st.selectbox("Internet Use", ['yes','no'])
        experience = st.slider("Farming Experience (years)", 1, 50, 10)
        climate    = st.slider("Climate Risk Score (hazards experienced)", 0, 5, 2)

    if st.button("Predict", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            'education level':       education,
            'gender':                gender,
            'age group':             age_group,
            'agricultural financing':financing,
            'phone ownership':       phone,
            'internet use':          internet,
            'farming experience':    float(experience),
            'climate_risk_score':    float(climate)
        }])
        input_df = input_df[[f for f in avail_features if f in input_df.columns]]

        prob = pipe.predict_proba(input_df)[0][1]
        pred = pipe.predict(input_df)[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Adoption Probability", f"{prob*100:.1f}%")
        m2.metric("Predicted Class", "High Adopter ✅" if pred else "Low Adopter ⚠️")
        m3.metric("Climate Vulnerability",
                  "🔴 High" if climate >= 3 else "🟡 Medium" if climate >= 1 else "🟢 Low")

        # Gauge bar
        fig, ax = plt.subplots(figsize=(7, 1))
        ax.barh([""], [prob], color="#2E8B57", height=0.5)
        ax.barh([""], [1-prob], left=[prob], color="#E8E8E8", height=0.5)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.text(prob/2, 0, f"{prob*100:.0f}%", ha='center', va='center',
                color='white', fontweight='bold', fontsize=13)
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False)
        ax.set_xlabel("Probability of High Adoption")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Recommended Action")
        if prob >= 0.70:
            st.success("✅ **High potential** — prioritise for advanced inputs and larger credit packages.")
        elif prob >= 0.45:
            st.warning("⚠️ **Moderate potential** — include in standard extension programme with monitoring.")
        else:
            st.info("📌 **Lower current likelihood** — consider targeted training on input benefits first.")
        if climate >= 3:
            st.error("🌧️ **High climate vulnerability** — prioritise for resilience interventions regardless of adoption score.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Scoring
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Score multiple farmers from a CSV")
    st.markdown(f"""
    Upload a CSV with these columns:
    `{', '.join(FEATURES)}`
    """)

    batch_file = st.file_uploader("Upload farmer CSV", type=["csv"], key="batch")

    if batch_file:
        batch_df = pd.read_csv(batch_file)
    else:
        st.info("No file uploaded — showing demo batch of 15 farmers.")
        batch_df = make_synthetic_data(15)[avail_features]

    st.write(f"**{len(batch_df):,} farmers loaded**")
    st.dataframe(batch_df.head(5), use_container_width=True)

    missing_cols = [c for c in avail_features if c not in batch_df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
    else:
        score_input = batch_df[[f for f in avail_features if f in batch_df.columns]]
        probs = pipe.predict_proba(score_input)[:, 1]
        preds = pipe.predict(score_input)

        batch_df = batch_df.copy()
        batch_df['adoption_probability'] = probs.round(3)
        batch_df['predicted_class']      = np.where(preds == 1, 'High Adopter', 'Low Adopter')
        batch_df['priority_tier']        = pd.cut(
            probs,
            bins=[0, 0.45, 0.70, 1.0],
            labels=['Standard', 'Medium Priority', 'High Priority']
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 High Priority",   (batch_df['priority_tier']=='High Priority').sum())
        c2.metric("🟡 Medium Priority", (batch_df['priority_tier']=='Medium Priority').sum())
        c3.metric("🟢 Standard",        (batch_df['priority_tier']=='Standard').sum())

        st.dataframe(
            batch_df.sort_values('adoption_probability', ascending=False),
            use_container_width=True
        )
        st.download_button(
            "⬇️ Download Scored CSV",
            batch_df.to_csv(index=False),
            file_name="farmers_scored.csv",
            mime="text/csv",
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Programme Summary Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Farmers", f"{len(df):,}")
    c2.metric("High Adopters", f"{df[TARGET].mean()*100:.1f}%")
    if 'climate_risk_score' in df.columns:
        c3.metric("Avg Climate Risk", f"{df['climate_risk_score'].mean():.2f} / 5")
    if 'agricultural financing' in df.columns:
        fin_rate = (df['agricultural financing'].map({'yes':1,'no':0}).fillna(0).mean())
        c4.metric("Financing Access", f"{fin_rate*100:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Adoption by education
    if 'education level' in df.columns:
        order = [e for e in ['no formal education','primary','secondary','tertiary']
                 if e in df['education level'].unique()]
        rates = df.groupby('education level')[TARGET].mean().mul(100).reindex(order)
        axes[0].bar(rates.index, rates.values, color='steelblue')
        axes[0].set_title('Adoption Rate by Education', fontweight='bold')
        axes[0].set_ylabel('High Adoption Rate (%)')
        axes[0].tick_params(axis='x', rotation=30)
        for i, v in enumerate(rates.values):
            axes[0].text(i, v+1, f'{v:.0f}%', ha='center', fontsize=9)

    # Adoption by financing
    if 'agricultural financing' in df.columns:
        fin_rates = df.groupby('agricultural financing')[TARGET].mean().mul(100)
        axes[1].bar(fin_rates.index, fin_rates.values, color=['#D16A5B','#5BA85B'])
        axes[1].set_title('Adoption Rate by Financing', fontweight='bold')
        axes[1].set_ylabel('High Adoption Rate (%)')
        for i, v in enumerate(fin_rates.values):
            axes[1].text(i, v+1, f'{v:.0f}%', ha='center', fontsize=9)

    # Climate risk distribution
    if 'climate_risk_score' in df.columns:
        axes[2].hist(df['climate_risk_score'], bins=6, color='tomato', edgecolor='white')
        axes[2].set_title('Climate Risk Distribution', fontweight='bold')
        axes[2].set_xlabel('Risk Score (0–5)')
        axes[2].set_ylabel('Number of Farmers')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Regional table
    if 'region' in df.columns:
        st.subheader("Adoption by Region")
        reg = df.groupby('region').agg(
            Farmers=(TARGET, 'count'),
            Adoption_Rate=(TARGET, lambda x: f"{x.mean()*100:.1f}%"),
            Avg_Climate_Risk=('climate_risk_score', lambda x: f"{x.mean():.2f}" if 'climate_risk_score' in df.columns else 'N/A')
        ).reset_index().sort_values('Farmers', ascending=False)
        st.dataframe(reg, use_container_width=True)

    # Hazard prevalence
    risk_cols = [c for c in ['losses-rain pattern','losses-drought','losses-heatwave',
                              'losses-storms','losses-mudslides'] if c in df.columns]
    if risk_cols:
        st.subheader("Climate Hazard Prevalence")
        hazard_labels = {
            'losses-rain pattern': 'Irregular Rainfall',
            'losses-drought':      'Drought',
            'losses-heatwave':     'Heatwave',
            'losses-storms':       'Storms',
            'losses-mudslides':    'Mudslides'
        }
        rates_h = df[risk_cols].mean().mul(100).rename(hazard_labels)
        fig2, ax2 = plt.subplots(figsize=(9, 3))
        colors = ['#E05C5C' if v > rates_h.median() else '#5C8AE0' for v in rates_h.values]
        bars = ax2.barh(rates_h.index, rates_h.values, color=colors)
        for bar, val in zip(bars, rates_h.values):
            ax2.text(val+0.5, bar.get_y()+bar.get_height()/2,
                     f'{val:.1f}%', va='center', fontsize=9)
        ax2.set_xlabel("Farmers Affected (%)")
        ax2.set_title("Prevalence of Climate Hazards", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Model Evaluation")

    # CV scores table
    st.markdown("#### Cross-Validation Results (5-Fold ROC-AUC)")
    cv_df = pd.DataFrame([
        {'Model': k, 'CV AUC': f"{v:.3f}",
         'Selected': '✅ Best' if k == best_name else ''}
        for k, v in cv_scores.items()
    ])
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

    # Train/test evaluation
    X_all = df[[f for f in avail_features if f in df.columns]]
    y_all = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
    )
    test_pipe = pipe.__class__(**pipe.get_params()) if False else pipe  # reuse fitted pipe
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Confusion Matrix")
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_te, y_pred,
            display_labels=['Low Adopter', 'High Adopter'],
            cmap='Blues', ax=ax3
        )
        ax3.set_title(f'{best_name}', fontweight='bold')
        st.pyplot(fig3)
        plt.close()

    with c2:
        st.markdown("#### ROC Curve")
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_te, y_proba, ax=ax4, name=best_name)
        ax4.plot([0,1],[0,1],'k--', label='Random')
        ax4.set_title('ROC Curve', fontweight='bold')
        ax4.legend()
        st.pyplot(fig4)
        plt.close()

    st.markdown(f"""
    **Test ROC-AUC:** `{roc_auc_score(y_te, y_proba):.3f}`

    ```
    {classification_report(y_te, y_pred, target_names=['Low Adopter','High Adopter'])}
    ```
    """)

    # Feature importance
    clf = pipe.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        imp = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        imp = np.abs(clf.coef_[0])
    else:
        imp = None

    if imp is not None:
        st.markdown("#### Feature Importance")
        imp_df = pd.DataFrame({
            'Feature': avail_features[:len(imp)],
            'Importance': imp
        }).sort_values('Importance', ascending=True)

        fig5, ax5 = plt.subplots(figsize=(8, 4))
        colors = ['#E05C5C' if v > imp_df['Importance'].median() else '#5C8AE0'
                  for v in imp_df['Importance']]
        ax5.barh(imp_df['Feature'], imp_df['Importance'], color=colors)
        ax5.set_title(f'Feature Importance – {best_name}', fontweight='bold')
        ax5.set_xlabel('Importance Score')
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Smart Grid ML Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
section[data-testid="stSidebar"] {width: 290px;}
.metric-card {padding: 1rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.25);}
.small {font-size: .88rem; color: rgba(120,120,120,1);}
</style>
""", unsafe_allow_html=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

@st.cache_data(show_spinner=False)
def load_data(file_bytes=None, file_name=None):
    if file_bytes is not None:
        from io import BytesIO
        return pd.read_csv(BytesIO(file_bytes))
    return pd.read_csv("capstone_smartgrid_20000.csv")


def is_binary(s: pd.Series) -> bool:
    v = s.dropna()
    if v.empty:
        return False
    return set(np.unique(v)).issubset({0, 1, 0.0, 1.0, True, False})


def base_preprocess(raw_df):
    df = raw_df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "month" not in df.columns:
        df["month"] = df["timestamp"].dt.month
    if "day" not in df.columns:
        df["day"] = df["timestamp"].dt.day
    if "year" not in df.columns:
        df["year"] = df["timestamp"].dt.year
    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    numeric_cols_current = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols_current:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    categorical_cols_current = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols_current:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    drop_for_model = [c for c in ["timestamp", "meter_id"] if c in df.columns]
    df_model = df.drop(columns=drop_for_model).copy()

    categorical_cols_model = df_model.select_dtypes(include=["object", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(
        df_model, columns=categorical_cols_model, drop_first=True, dtype=int
    )

    possible_targets = [
        "next_hour_consumption_kwh",
        "high_usage_flag",
        "anomaly_flag",
        "outage_risk_score"
    ]

    X_prepared = df_encoded.copy()
    binary_cols = [
        c for c in X_prepared.columns
        if X_prepared[c].dropna().isin([0, 1]).all()
    ]
    possible_target_cols = [c for c in possible_targets if c in X_prepared.columns]
    scale_cols = [
        c for c in X_prepared.select_dtypes(include=np.number).columns
        if c not in binary_cols and c not in possible_target_cols
    ]
    scaler = StandardScaler()
    if scale_cols:
        X_prepared[scale_cols] = scaler.fit_transform(X_prepared[scale_cols])

    return df, df_model, df_encoded, X_prepared, possible_targets, scale_cols, binary_cols


def train_logistic(df_encoded, raw_df):
    target_col = "anomaly_flag"
    y = df_encoded[target_col].astype(int)
    possible_targets = [
        "next_hour_consumption_kwh", "high_usage_flag", "anomaly_flag", "outage_risk_score"
    ]
    drop_cols = [c for c in possible_targets if c in df_encoded.columns]
    for extra in ["timestamp", "meter_id"]:
        if extra in df_encoded.columns:
            drop_cols.append(extra)
    X = df_encoded.drop(columns=drop_cols)
    const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const_cols:
        X = X.drop(columns=const_cols)

    order = np.argsort(pd.to_datetime(raw_df["timestamp"], errors="coerce").values)
    cut = int(len(order) * (1 - TEST_SIZE))
    tr_idx, te_idx = order[:cut], order[cut:]
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]

    binary_cols = [c for c in X_train.columns if is_binary(X_train[c])]
    cont_cols = [c for c in X_train.columns if c not in binary_cols]
    preprocessor = ColumnTransformer([
        ("scale", StandardScaler(), cont_cols),
        ("pass", "passthrough", binary_cols)
    ], remainder="drop", verbose_feature_names_out=False)

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(
            solver="liblinear", C=1.0, class_weight="balanced",
            max_iter=2000, random_state=RANDOM_STATE
        ))
    ])
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE).fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC": average_precision_score(y_test, proba)
    }
    return {
        "model": model, "X_test": X_test, "y_test": y_test, "pred": pred,
        "proba": proba, "dummy_accuracy": accuracy_score(y_test, dummy_pred),
        "dummy_recall": recall_score(y_test, dummy_pred, zero_division=0),
        "metrics": metrics
    }


def train_ridge(raw_df):
    target = "next_hour_consumption_kwh"
    model_df = raw_df.copy()
    model_df["timestamp"] = pd.to_datetime(model_df["timestamp"], errors="coerce")
    model_df = model_df.sort_values("timestamp").reset_index(drop=True)
    model_df = model_df.dropna(subset=[target, "timestamp"]).reset_index(drop=True)

    drop_cols = [c for c in [target, "timestamp", "meter_id", "high_usage_flag", "anomaly_flag", "outage_risk_score"] if c in model_df.columns]
    X_raw = model_df.drop(columns=drop_cols)
    y = model_df[target]

    numeric_cols = X_raw.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    split_index = int(len(X_raw) * 0.80)
    X_train_raw = X_raw.iloc[:split_index].copy()
    X_test_raw = X_raw.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    pipe = Pipeline([("ridge", Ridge())])
    grid = GridSearchCV(
        pipe,
        {"ridge__alpha": [0.01, 0.1, 1, 10, 100]},
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_test_pred),
        "MSE": mean_squared_error(y_test, y_test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "R²": r2_score(y_test, y_test_pred)
    }
    return {
        "model": best_model, "best_alpha": grid.best_params_["ridge__alpha"],
        "best_cv_rmse": -grid.best_score_, "y_train": y_train,
        "y_test": y_test, "y_train_pred": y_train_pred, "y_test_pred": y_test_pred,
        "metrics": metrics
    }


def train_rf(df_encoded):
    targets_to_drop = ["next_hour_consumption_kwh", "high_usage_flag", "anomaly_flag", "outage_risk_score"]
    X = df_encoded.drop(columns=[c for c in targets_to_drop if c in df_encoded.columns])
    y = df_encoded["anomaly_flag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    train_pred = model.predict(X_train)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC": average_precision_score(y_test, proba)
    }
    imp = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False).head(10)
    return {"model": model, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "pred": pred, "proba": proba, "train_f1": f1_score(y_train, train_pred, zero_division=0), "metrics": metrics, "importance": imp}


def train_xgb(X_prepared):
    X = X_prepared.drop(columns=["high_usage_flag", "consumption_kwh"])
    y = X_prepared["high_usage_flag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=42, eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC": average_precision_score(y_test, proba)
    }
    return {"model": model, "X_test": X_test, "y_test": y_test, "pred": pred, "proba": proba, "metrics": metrics}


def metric_table(metrics):
    return pd.DataFrame({"Metric": list(metrics.keys()), "Score": [round(v, 4) for v in metrics.values()]})


st.title("⚡ Smart Grid — Notebook → Streamlit")
st.caption("The Streamlit app follows the same project flow as SmartGrid.ipynb: EDA → preprocessing → four models → evaluation.")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload the CSV (optional)", type=["csv"])
    page = st.radio(
        "Navigate",
        [
            "Overview", "EDA", "Data Processing",
            "Logistic Regression", "Ridge Regression",
            "Random Forest", "XGBoost", "Model Summary"
        ]
    )
    st.divider()
    run_all = st.button("Run / refresh all models", use_container_width=True)

file_bytes = uploaded.getvalue() if uploaded else None
file_name = uploaded.name if uploaded else None
raw_df = load_data(file_bytes, file_name)

if run_all:
    st.cache_data.clear()
    st.session_state.pop("artifacts", None)
    st.rerun()

if "artifacts" not in st.session_state:
    st.session_state.artifacts = {}

if page in ["Overview", "EDA", "Data Processing", "Logistic Regression", "Ridge Regression", "Random Forest", "XGBoost", "Model Summary"]:
    try:
        df_processed, df_model, df_encoded, X_prepared, possible_targets, scale_cols, binary_cols = base_preprocess(raw_df)
        st.session_state.artifacts.update({
            "df_processed": df_processed, "df_model": df_model, "df_encoded": df_encoded,
            "X_prepared": X_prepared, "possible_targets": possible_targets,
            "scale_cols": scale_cols, "binary_cols": binary_cols
        })
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

if page == "Overview":
    st.subheader("Project Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(raw_df):,}")
    c2.metric("Columns", f"{raw_df.shape[1]}")
    c3.metric("Anomalies", f"{int(raw_df['anomaly_flag'].sum()):,}")
    c4.metric("High usage", f"{int(raw_df['high_usage_flag'].sum()):,}")
    st.markdown("### Original Notebook Flow")
    st.markdown("**0–15:** Data inspection, missing values, duplicates, outliers, correlation, timestamp features, imputation, removing identifiers, One-Hot Encoding, target inspection and scaling.")
    st.markdown("**16–17:** Logistic Regression baseline and imbalanced-class evaluation.")
    st.markdown("**18–19:** Ridge Regression for next-hour consumption, alpha tuning and residual analysis.")
    st.markdown("**20:** Random Forest for anomaly classification and feature importance.")
    st.markdown("**21:** XGBoost for high-usage classification.")

elif page == "EDA":
    st.subheader("EDA")
    tabs = st.tabs(["Basic inspection", "Missing values", "Duplicates", "Targets", "Outliers", "Correlation"])
    with tabs[0]:
        st.write("Shape:", raw_df.shape)
        st.dataframe(raw_df.head())
        st.dataframe(raw_df.dtypes.astype(str).rename("dtype").to_frame())
        st.dataframe(raw_df.describe(include="all").T)
    with tabs[1]:
        miss = raw_df.isnull().sum().sort_values(ascending=False)
        miss = miss[miss > 0]
        st.dataframe(pd.DataFrame({"Missing Values": miss, "Missing %": (miss / len(raw_df) * 100).round(2)}))
    with tabs[2]:
        st.metric("Duplicate rows", int(raw_df.duplicated().sum()))
    with tabs[3]:
        st.write("Target distributions from the notebook:")
        cols = [c for c in ["high_usage_flag", "anomaly_flag"] if c in raw_df.columns]
        st.dataframe(pd.DataFrame({c: raw_df[c].value_counts().sort_index() for c in cols}).fillna(0).astype(int))
    with tabs[4]:
        numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
        outlier_rows = []
        for col in numeric_cols:
            s = raw_df[col].dropna()
            if s.empty: continue
            q1, q3 = s.quantile(.25), s.quantile(.75)
            iqr = q3 - q1
            if iqr == 0:
                count = int((s != q1).sum())
            else:
                lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
                count = int(((s < lo) | (s > hi)).sum())
            outlier_rows.append([col, count, round(count / len(raw_df) * 100, 2)])
        out = pd.DataFrame(outlier_rows, columns=["Column", "Outliers", "Outlier %"]).sort_values("Outliers", ascending=False)
        st.dataframe(out)
        chosen = st.selectbox("Boxplot column", numeric_cols)
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.boxplot(x=raw_df[chosen], ax=ax)
        ax.set_title(f"Boxplot — {chosen}")
        st.pyplot(fig, clear_figure=True)
    with tabs[5]:
        numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
        corr = raw_df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig, clear_figure=True)
        corr_pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
        corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
        st.dataframe(corr_pairs.reindex(corr_pairs["Correlation"].abs().sort_values(ascending=False).index).head(20))

elif page == "Data Processing":
    st.subheader("Data Processing")
    c1, c2, c3 = st.columns(3)
    c1.metric("After One-Hot Encoding", f"{df_encoded.shape[1]} features")
    c2.metric("After final preparation", f"{X_prepared.shape[1]} columns")
    c3.metric("Remaining missing", int(X_prepared.isna().sum().sum()))
    st.markdown("### Processing steps")
    st.write("1. Timestamp → month/day/year + cyclic hour/day-of-week features")
    st.write("2. Numeric missing values → median")
    st.write("3. Categorical missing values → mode")
    st.write("4. Remove raw timestamp and meter_id from model inputs")
    st.write("5. One-Hot Encoding with `drop_first=True`")
    st.write("6. StandardScaler on non-binary numeric features; binary columns remain 0/1")
    st.markdown("### Scaled columns")
    st.dataframe(pd.DataFrame({"Scaled": scale_cols}))
    st.markdown("### Final prepared data")
    st.dataframe(X_prepared.head())

elif page == "Logistic Regression":
    st.subheader("Logistic Regression — Anomaly Detection")
    if "logistic" not in st.session_state.artifacts:
        with st.spinner("Training Logistic Regression using the notebook logic..."):
            st.session_state.artifacts["logistic"] = train_logistic(df_encoded, df_processed)
    r = st.session_state.artifacts["logistic"]
    st.info("Target: `anomaly_flag` | Time-based 80/20 split | class_weight='balanced'")
    st.dataframe(metric_table(r["metrics"]), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Dummy accuracy", f"{r['dummy_accuracy']:.4f}")
    c2.metric("Dummy recall", f"{r['dummy_recall']:.4f}")
    cm = confusion_matrix(r["y_test"], r["pred"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Logistic Regression — Confusion Matrix")
    st.pyplot(fig, clear_figure=True)
    st.code(classification_report(r["y_test"], r["pred"], zero_division=0))

elif page == "Ridge Regression":
    st.subheader("Ridge Regression — Next-Hour Consumption")
    if "ridge" not in st.session_state.artifacts:
        with st.spinner("Training Ridge using the notebook logic..."):
            st.session_state.artifacts["ridge"] = train_ridge(raw_df)
    r = st.session_state.artifacts["ridge"]
    c1, c2 = st.columns(2)
    c1.metric("Best alpha", str(r["best_alpha"]))
    c2.metric("Test R²", f"{r['metrics']['R²']:.4f}")
    st.dataframe(metric_table(r["metrics"]), use_container_width=True)
    st.markdown("### Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(r["y_test"], r["y_test_pred"], alpha=.45)
    lo = min(r["y_test"].min(), r["y_test_pred"].min()); hi = max(r["y_test"].max(), r["y_test_pred"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Actual Next-Hour Consumption"); ax.set_ylabel("Predicted Next-Hour Consumption")
    st.pyplot(fig, clear_figure=True)
    errors = r["y_test"] - r["y_test_pred"]
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(errors, bins=40); ax.set_title("Prediction Error Distribution"); ax.set_xlabel("Actual - Predicted"); st.pyplot(fig, clear_figure=True)
    with c2:
        fig, ax = plt.subplots(figsize=(7, 4)); ax.scatter(r["y_test_pred"], errors, alpha=.45); ax.axhline(0, linestyle="--"); ax.set_title("Residuals vs Predictions"); st.pyplot(fig, clear_figure=True)
    st.write(f"Mean error: {errors.mean():.4f} | Median error: {errors.median():.4f}")

elif page == "Random Forest":
    st.subheader("Random Forest — Anomaly Detection")
    if "rf" not in st.session_state.artifacts:
        with st.spinner("Training Random Forest using the notebook logic..."):
            st.session_state.artifacts["rf"] = train_rf(df_encoded)
    r = st.session_state.artifacts["rf"]
    st.info("Target: `anomaly_flag` | stratified 80/20 split | no scaling")
    st.dataframe(metric_table(r["metrics"]), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Train F1", f"{r['train_f1']:.4f}")
    c2.metric("Test F1", f"{r['metrics']['F1']:.4f}")
    c3.metric("F1 Gap", f"{r['train_f1'] - r['metrics']['F1']:.4f}")
    cm = confusion_matrix(r["y_test"], r["pred"])
    fig, ax = plt.subplots(figsize=(5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax); ax.set_title("Random Forest — Confusion Matrix"); st.pyplot(fig, clear_figure=True)
    st.markdown("### Top 10 Feature Importances")
    st.dataframe(r["importance"], use_container_width=True)
    fig, ax = plt.subplots(figsize=(9, 5)); ax.barh(r["importance"]["Feature"][::-1], r["importance"]["Importance"][::-1]); ax.set_xlabel("Importance"); st.pyplot(fig, clear_figure=True)

elif page == "XGBoost":
    st.subheader("XGBoost — High Usage Classification")
    if "xgb" not in st.session_state.artifacts:
        with st.spinner("Training XGBoost using the notebook logic..."):
            st.session_state.artifacts["xgb"] = train_xgb(X_prepared)
    r = st.session_state.artifacts["xgb"]
    st.info("Target: `high_usage_flag` | 0 = Normal Usage | 1 = High Usage")
    st.dataframe(metric_table(r["metrics"]), use_container_width=True)
    cm = confusion_matrix(r["y_test"], r["pred"])
    fig, ax = plt.subplots(figsize=(5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax); ax.set_title("XGBoost — Confusion Matrix"); st.pyplot(fig, clear_figure=True)
    st.code(classification_report(r["y_test"], r["pred"], zero_division=0))

elif page == "Model Summary":
    st.subheader("Four-Model Summary")
    results = []
    with st.spinner("Preparing model results..."):
        if "logistic" not in st.session_state.artifacts:
            st.session_state.artifacts["logistic"] = train_logistic(df_encoded, df_processed)
        if "ridge" not in st.session_state.artifacts:
            st.session_state.artifacts["ridge"] = train_ridge(raw_df)
        if "rf" not in st.session_state.artifacts:
            st.session_state.artifacts["rf"] = train_rf(df_encoded)
        if "xgb" not in st.session_state.artifacts:
            st.session_state.artifacts["xgb"] = train_xgb(X_prepared)
    lr = st.session_state.artifacts["logistic"]
    rg = st.session_state.artifacts["ridge"]
    rf = st.session_state.artifacts["rf"]
    xb = st.session_state.artifacts["xgb"]
    results = [
        {"Model":"Logistic Regression","Task":"Classification","Target":"anomaly_flag",**lr["metrics"]},
        {"Model":"Ridge Regression","Task":"Regression","Target":"next_hour_consumption_kwh",**rg["metrics"]},
        {"Model":"Random Forest","Task":"Classification","Target":"anomaly_flag",**rf["metrics"]},
        {"Model":"XGBoost","Task":"Classification","Target":"high_usage_flag",**xb["metrics"]},
    ]
    summary = pd.DataFrame(results)
    st.dataframe(summary.round(4), use_container_width=True)
    st.warning("Classification models in the notebook do not share the same target: Logistic/Random Forest use anomaly_flag, while XGBoost uses high_usage_flag. Ridge is a separate regression task, so its MAE/RMSE/R² should not be compared with classification metrics.")

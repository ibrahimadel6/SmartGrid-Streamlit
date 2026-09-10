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
    classification_report, mean_absolute_error, mean_squared_error,
    r2_score
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
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
section[data-testid="stSidebar"] {
    width: 290px;
}
</style>
""", unsafe_allow_html=True)


RANDOM_STATE = 42
TEST_SIZE = 0.20


@st.cache_data(show_spinner=False)
def load_data(file_bytes=None):
    if file_bytes is not None:
        from io import BytesIO
        df = pd.read_csv(BytesIO(file_bytes))
    else:
        df = pd.read_csv("capstone_smartgrid_20000.csv")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype(np.int32)

    return df


def is_binary(series):
    values = series.dropna()

    if values.empty:
        return False

    unique_values = set(values.unique())

    return unique_values.issubset({
        0, 1, 0.0, 1.0, True, False
    })


@st.cache_data(show_spinner=False)
def prepare_data(raw_df):

    df = raw_df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if "month" not in df.columns:
        df["month"] = df["timestamp"].dt.month

    if "day" not in df.columns:
        df["day"] = df["timestamp"].dt.day

    if "year" not in df.columns:
        df["year"] = df["timestamp"].dt.year

    if "hour" in df.columns:
        df["hour_sin"] = np.sin(
            2 * np.pi * df["hour"] / 24
        ).astype(np.float32)

        df["hour_cos"] = np.cos(
            2 * np.pi * df["hour"] / 24
        ).astype(np.float32)

    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(
            2 * np.pi * df["day_of_week"] / 7
        ).astype(np.float32)

        df["dow_cos"] = np.cos(
            2 * np.pi * df["day_of_week"] / 7
        ).astype(np.float32)

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().any():
            median_value = df[col].median()

            if pd.isna(median_value):
                median_value = 0

            df[col] = df[col].fillna(median_value)

    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_values = df[col].mode()

            if len(mode_values) > 0:
                df[col] = df[col].fillna(mode_values.iloc[0])
            else:
                df[col] = df[col].fillna("Unknown")

    drop_columns = [
        c for c in ["timestamp", "meter_id"]
        if c in df.columns
    ]

    df_model = df.drop(
        columns=drop_columns
    )

    categorical_model_cols = df_model.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    df_encoded = pd.get_dummies(
        df_model,
        columns=categorical_model_cols,
        drop_first=True,
        dtype=np.int8
    )

    numeric_encoded = df_encoded.select_dtypes(
        include=np.number
    ).columns.tolist()

    target_columns = [
        "next_hour_consumption_kwh",
        "high_usage_flag",
        "anomaly_flag",
        "outage_risk_score"
    ]

    binary_columns = [
        c for c in numeric_encoded
        if c not in target_columns
        and is_binary(df_encoded[c])
    ]

    scale_columns = [
        c for c in numeric_encoded
        if c not in binary_columns
        and c not in target_columns
    ]

    x_prepared = df_encoded.copy()

    if scale_columns:

        scaler = StandardScaler()

        x_prepared[scale_columns] = scaler.fit_transform(
            x_prepared[scale_columns]
        ).astype(np.float32)

    for col in x_prepared.select_dtypes(
        include=["float64"]
    ).columns:

        x_prepared[col] = x_prepared[col].astype(
            np.float32
        )

    return (
        df_encoded,
        x_prepared,
        scale_columns,
        binary_columns
    )


def classification_metrics(y_true, predictions, probabilities):

    metrics = {
        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        )
    }

    try:
        metrics["ROC-AUC"] = roc_auc_score(
            y_true,
            probabilities
        )
    except:
        metrics["ROC-AUC"] = 0.0

    try:
        metrics["PR-AUC"] = average_precision_score(
            y_true,
            probabilities
        )
    except:
        metrics["PR-AUC"] = 0.0

    return metrics


@st.cache_data(show_spinner=False)
def train_logistic(raw_df, df_encoded):

    target = "anomaly_flag"

    y = df_encoded[target].astype(np.int8)

    target_columns = [
        "next_hour_consumption_kwh",
        "high_usage_flag",
        "anomaly_flag",
        "outage_risk_score"
    ]

    drop_columns = [
        c for c in target_columns
        if c in df_encoded.columns
    ]

    x = df_encoded.drop(
        columns=drop_columns
    )

    constant_columns = [
        c for c in x.columns
        if x[c].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        x = x.drop(
            columns=constant_columns
        )

    timestamps = pd.to_datetime(
        raw_df["timestamp"],
        errors="coerce"
    )

    order = np.argsort(
        timestamps.values
    )

    split_index = int(
        len(order) * (1 - TEST_SIZE)
    )

    train_indices = order[:split_index]
    test_indices = order[split_index:]

    x_train = x.iloc[train_indices]
    x_test = x.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    binary_columns = [
        c for c in x_train.columns
        if is_binary(x_train[c])
    ]

    continuous_columns = [
        c for c in x_train.columns
        if c not in binary_columns
    ]

    preprocessor = ColumnTransformer(
        [
            (
                "scale",
                StandardScaler(),
                continuous_columns
            ),
            (
                "pass",
                "passthrough",
                binary_columns
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    model = Pipeline(
        [
            (
                "prep",
                preprocessor
            ),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    model.fit(
        x_train,
        y_train
    )

    probabilities = model.predict_proba(
        x_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(np.int8)

    dummy = DummyClassifier(
        strategy="most_frequent",
        random_state=RANDOM_STATE
    )

    dummy.fit(
        x_train,
        y_train
    )

    dummy_predictions = dummy.predict(
        x_test
    )

    metrics = classification_metrics(
        y_test,
        predictions,
        probabilities
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    return {
        "y_test": y_test.to_numpy(),
        "pred": predictions,
        "proba": probabilities,
        "metrics": metrics,
        "dummy_accuracy": accuracy_score(
            y_test,
            dummy_predictions
        ),
        "dummy_recall": recall_score(
            y_test,
            dummy_predictions,
            zero_division=0
        ),
        "report": report
    }


@st.cache_data(show_spinner=False)
def train_ridge(raw_df):

    target = "next_hour_consumption_kwh"

    df = raw_df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    df = df.dropna(
        subset=[
            target,
            "timestamp"
        ]
    ).reset_index(drop=True)

    drop_columns = [
        target,
        "timestamp",
        "meter_id",
        "high_usage_flag",
        "anomaly_flag",
        "outage_risk_score"
    ]

    drop_columns = [
        c for c in drop_columns
        if c in df.columns
    ]

    x_raw = df.drop(
        columns=drop_columns
    )

    y = df[target].astype(
        np.float32
    )

    numeric_columns = x_raw.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = x_raw.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    split_index = int(
        len(x_raw) * 0.80
    )

    x_train_raw = x_raw.iloc[
        :split_index
    ]

    x_test_raw = x_raw.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    pipe = Pipeline(
        [
            (
                "prep",
                preprocessor
            ),
            (
                "ridge",
                Ridge(
                    solver="lsqr"
                )
            )
        ]
    )

    grid = GridSearchCV(
        pipe,
        {
            "ridge__alpha": [
                0.01,
                0.1,
                1,
                10,
                100
            ]
        },
        cv=TimeSeriesSplit(
            n_splits=5
        ),
        scoring="neg_root_mean_squared_error",
        n_jobs=1
    )

    grid.fit(
        x_train_raw,
        y_train
    )

    best_model = grid.best_estimator_

    y_test_pred = best_model.predict(
        x_test_raw
    )

    metrics = {
        "MAE": mean_absolute_error(
            y_test,
            y_test_pred
        ),
        "MSE": mean_squared_error(
            y_test,
            y_test_pred
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_test,
                y_test_pred
            )
        ),
        "R²": r2_score(
            y_test,
            y_test_pred
        )
    }

    return {
        "y_test": y_test.to_numpy(),
        "y_test_pred": y_test_pred,
        "metrics": metrics,
        "best_alpha": grid.best_params_[
            "ridge__alpha"
        ],
        "best_cv_rmse": -grid.best_score_
    }


@st.cache_data(show_spinner=False)
def train_rf(df_encoded):

    target = "anomaly_flag"

    drop_columns = [
        "next_hour_consumption_kwh",
        "high_usage_flag",
        "anomaly_flag",
        "outage_risk_score"
    ]

    drop_columns = [
        c for c in drop_columns
        if c in df_encoded.columns
    ]

    x = df_encoded.drop(
        columns=drop_columns
    )

    y = df_encoded[target].astype(
        np.int8
    )

    x = x.astype(
        np.float32
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    model.fit(
        x_train,
        y_train
    )

    predictions = model.predict(
        x_test
    )

    probabilities = model.predict_proba(
        x_test
    )[:, 1]

    train_predictions = model.predict(
        x_train
    )

    metrics = classification_metrics(
        y_test,
        predictions,
        probabilities
    )

    importance = pd.DataFrame(
        {
            "Feature": x.columns,
            "Importance": model.feature_importances_
        }
    ).sort_values(
        "Importance",
        ascending=False
    ).head(10)

    return {
        "y_test": y_test.to_numpy(),
        "pred": predictions,
        "proba": probabilities,
        "metrics": metrics,
        "train_f1": f1_score(
            y_train,
            train_predictions,
            zero_division=0
        ),
        "importance": importance
    }


@st.cache_data(show_spinner=False)
def train_xgb(x_prepared):

    target = "high_usage_flag"

    drop_columns = [
        "high_usage_flag",
        "consumption_kwh"
    ]

    drop_columns = [
        c for c in drop_columns
        if c in x_prepared.columns
    ]

    x = x_prepared.drop(
        columns=drop_columns
    )

    y = x_prepared[target].astype(
        np.int8
    )

    x = x.astype(
        np.float32
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=80,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        max_bin=64,
        tree_method="hist",
        n_jobs=1,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )

    model.fit(
        x_train,
        y_train
    )

    predictions = model.predict(
        x_test
    )

    probabilities = model.predict_proba(
        x_test
    )[:, 1]

    metrics = classification_metrics(
        y_test,
        predictions,
        probabilities
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    return {
        "y_test": y_test.to_numpy(),
        "pred": predictions,
        "proba": probabilities,
        "metrics": metrics,
        "report": report
    }


def metric_table(metrics):

    return pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Score": [
                round(
                    float(value),
                    4
                )
                for value in metrics.values()
            ]
        }
    )


st.title(
    "⚡ Smart Grid — Notebook → Streamlit"
)

st.caption(
    "Smart Grid ML Dashboard: EDA → preprocessing → four models → evaluation."
)


with st.sidebar:

    st.header("Controls")

    uploaded = st.file_uploader(
        "Upload the CSV (optional)",
        type=["csv"]
    )

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "EDA",
            "Data Processing",
            "Logistic Regression",
            "Ridge Regression",
            "Random Forest",
            "XGBoost",
            "Model Summary"
        ]
    )

    st.divider()

    run_all = st.button(
        "Clear cache and refresh",
        use_container_width=True
    )


file_bytes = uploaded.getvalue() if uploaded else None

raw_df = load_data(
    file_bytes
)


if run_all:
    st.cache_data.clear()
    st.rerun()


if page == "Overview":

    st.subheader(
        "Project Overview"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        f"{len(raw_df):,}"
    )

    c2.metric(
        "Columns",
        f"{raw_df.shape[1]}"
    )

    if "anomaly_flag" in raw_df.columns:
        c3.metric(
            "Anomalies",
            f"{int(raw_df['anomaly_flag'].sum()):,}"
        )
    else:
        c3.metric(
            "Anomalies",
            "N/A"
        )

    if "high_usage_flag" in raw_df.columns:
        c4.metric(
            "High usage",
            f"{int(raw_df['high_usage_flag'].sum()):,}"
        )
    else:
        c4.metric(
            "High usage",
            "N/A"
        )

    st.markdown(
        "### Original Notebook Flow"
    )

    st.markdown(
        "**0–15:** Data inspection, missing values, duplicates, outliers, correlation, timestamp features, imputation, identifiers removal, One-Hot Encoding and scaling."
    )

    st.markdown(
        "**16–17:** Logistic Regression baseline and imbalanced-class evaluation."
    )

    st.markdown(
        "**18–19:** Ridge Regression for next-hour consumption, alpha tuning and residual analysis."
    )

    st.markdown(
        "**20:** Random Forest for anomaly classification and feature importance."
    )

    st.markdown(
        "**21:** XGBoost for high-usage classification."
    )


elif page == "EDA":

    st.subheader(
        "EDA"
    )

    tabs = st.tabs(
        [
            "Basic inspection",
            "Missing values",
            "Duplicates",
            "Targets",
            "Outliers",
            "Correlation"
        ]
    )

    with tabs[0]:

        st.write(
            "Shape:",
            raw_df.shape
        )

        st.dataframe(
            raw_df.head(),
            use_container_width=True
        )

        st.dataframe(
            raw_df.dtypes.astype(
                str
            ).rename(
                "dtype"
            ).to_frame(),
            use_container_width=True
        )

        st.dataframe(
            raw_df.describe(
                include="all"
            ).T,
            use_container_width=True
        )

    with tabs[1]:

        missing = raw_df.isnull().sum()

        missing = missing[
            missing > 0
        ].sort_values(
            ascending=False
        )

        if len(missing) == 0:

            st.success(
                "No missing values found."
            )

        else:

            missing_table = pd.DataFrame(
                {
                    "Missing Values": missing,
                    "Missing %": (
                        missing /
                        len(raw_df) *
                        100
                    ).round(2)
                }
            )

            st.dataframe(
                missing_table,
                use_container_width=True
            )

    with tabs[2]:

        st.metric(
            "Duplicate rows",
            int(
                raw_df.duplicated().sum()
            )
        )

    with tabs[3]:

        target_columns = [
            c for c in [
                "high_usage_flag",
                "anomaly_flag"
            ]
            if c in raw_df.columns
        ]

        if target_columns:

            target_table = pd.DataFrame(
                {
                    c: raw_df[c]
                    .value_counts()
                    .sort_index()
                    for c in target_columns
                }
            ).fillna(
                0
            ).astype(
                int
            )

            st.dataframe(
                target_table,
                use_container_width=True
            )

    with tabs[4]:

        numeric_columns = raw_df.select_dtypes(
            include=np.number
        ).columns.tolist()

        outlier_rows = []

        for col in numeric_columns:

            series = raw_df[col].dropna()

            if series.empty:
                continue

            q1 = series.quantile(
                0.25
            )

            q3 = series.quantile(
                0.75
            )

            iqr = q3 - q1

            if iqr == 0:

                count = int(
                    (series != q1).sum()
                )

            else:

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                count = int(
                    (
                        (series < lower) |
                        (series > upper)
                    ).sum()
                )

            outlier_rows.append(
                [
                    col,
                    count,
                    round(
                        count /
                        len(raw_df) *
                        100,
                        2
                    )
                ]
            )

        outlier_table = pd.DataFrame(
            outlier_rows,
            columns=[
                "Column",
                "Outliers",
                "Outlier %"
            ]
        ).sort_values(
            "Outliers",
            ascending=False
        )

        st.dataframe(
            outlier_table,
            use_container_width=True
        )

        chosen = st.selectbox(
            "Boxplot column",
            numeric_columns
        )

        fig, ax = plt.subplots(
            figsize=(9, 4)
        )

        sns.boxplot(
            x=raw_df[chosen],
            ax=ax
        )

        ax.set_title(
            f"Boxplot — {chosen}"
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

    with tabs[5]:

        numeric_columns = raw_df.select_dtypes(
            include=np.number
        ).columns.tolist()

        corr = raw_df[
            numeric_columns
        ].corr()

        fig, ax = plt.subplots(
            figsize=(12, 8)
        )

        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            ax=ax
        )

        ax.set_title(
            "Correlation Heatmap"
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

        corr_pairs = corr.where(
            np.triu(
                np.ones(
                    corr.shape
                ),
                k=1
            ).astype(bool)
        ).stack().reset_index()

        corr_pairs.columns = [
            "Feature 1",
            "Feature 2",
            "Correlation"
        ]

        corr_pairs = corr_pairs.reindex(
            corr_pairs["Correlation"]
            .abs()
            .sort_values(
                ascending=False
            )
            .index
        )

        st.dataframe(
            corr_pairs.head(20),
            use_container_width=True
        )


elif page == "Data Processing":

    st.subheader(
        "Data Processing"
    )

    (
        df_encoded,
        x_prepared,
        scale_columns,
        binary_columns
    ) = prepare_data(
        raw_df
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "After One-Hot Encoding",
        f"{df_encoded.shape[1]} features"
    )

    c2.metric(
        "After final preparation",
        f"{x_prepared.shape[1]} columns"
    )

    c3.metric(
        "Remaining missing",
        int(
            x_prepared.isna().sum().sum()
        )
    )

    st.markdown(
        "### Processing steps"
    )

    st.write(
        "1. Timestamp → month/day/year + cyclic hour/day-of-week features"
    )

    st.write(
        "2. Numeric missing values → median"
    )

    st.write(
        "3. Categorical missing values → mode"
    )

    st.write(
        "4. Remove timestamp and meter_id from model inputs"
    )

    st.write(
        "5. One-Hot Encoding with drop_first=True"
    )

    st.write(
        "6. StandardScaler on non-binary numeric features"
    )

    st.markdown(
        "### Scaled columns"
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Scaled": scale_columns
            }
        ),
        use_container_width=True
    )

    st.markdown(
        "### Final prepared data"
    )

    st.dataframe(
        x_prepared.head(),
        use_container_width=True
    )


elif page == "Logistic Regression":

    st.subheader(
        "Logistic Regression — Anomaly Detection"
    )

    (
        df_encoded,
        x_prepared,
        scale_columns,
        binary_columns
    ) = prepare_data(
        raw_df
    )

    with st.spinner(
        "Training Logistic Regression..."
    ):

        result = train_logistic(
            raw_df,
            df_encoded
        )

    st.info(
        "Target: anomaly_flag | Time-based 80/20 split | class_weight='balanced'"
    )

    st.dataframe(
        metric_table(
            result["metrics"]
        ),
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Dummy accuracy",
        f"{result['dummy_accuracy']:.4f}"
    )

    c2.metric(
        "Dummy recall",
        f"{result['dummy_recall']:.4f}"
    )

    cm = confusion_matrix(
        result["y_test"],
        result["pred"]
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        "Logistic Regression — Confusion Matrix"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)

    st.code(
        result["report"]
    )


elif page == "Ridge Regression":

    st.subheader(
        "Ridge Regression — Next-Hour Consumption"
    )

    with st.spinner(
        "Training Ridge Regression..."
    ):

        result = train_ridge(
            raw_df
        )

    c1, c2 = st.columns(2)

    c1.metric(
        "Best alpha",
        str(
            result["best_alpha"]
        )
    )

    c2.metric(
        "Test R²",
        f"{result['metrics']['R²']:.4f}"
    )

    st.dataframe(
        metric_table(
            result["metrics"]
        ),
        use_container_width=True
    )

    st.markdown(
        "### Actual vs Predicted"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.scatter(
        result["y_test"],
        result["y_test_pred"],
        alpha=0.45
    )

    low_value = min(
        result["y_test"].min(),
        result["y_test_pred"].min()
    )

    high_value = max(
        result["y_test"].max(),
        result["y_test_pred"].max()
    )

    ax.plot(
        [low_value, high_value],
        [low_value, high_value],
        linestyle="--"
    )

    ax.set_xlabel(
        "Actual Next-Hour Consumption"
    )

    ax.set_ylabel(
        "Predicted Next-Hour Consumption"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)

    errors = (
        result["y_test"] -
        result["y_test_pred"]
    )

    c1, c2 = st.columns(2)

    with c1:

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.hist(
            errors,
            bins=40
        )

        ax.set_title(
            "Prediction Error Distribution"
        )

        ax.set_xlabel(
            "Actual - Predicted"
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

    with c2:

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        ax.scatter(
            result["y_test_pred"],
            errors,
            alpha=0.45
        )

        ax.axhline(
            0,
            linestyle="--"
        )

        ax.set_title(
            "Residuals vs Predictions"
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

    st.write(
        f"Mean error: {errors.mean():.4f} | "
        f"Median error: {np.median(errors):.4f}"
    )


elif page == "Random Forest":

    st.subheader(
        "Random Forest — Anomaly Detection"
    )

    (
        df_encoded,
        x_prepared,
        scale_columns,
        binary_columns
    ) = prepare_data(
        raw_df
    )

    with st.spinner(
        "Training Random Forest..."
    ):

        result = train_rf(
            df_encoded
        )

    st.info(
        "Target: anomaly_flag | stratified 80/20 split | no scaling"
    )

    st.dataframe(
        metric_table(
            result["metrics"]
        ),
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Train F1",
        f"{result['train_f1']:.4f}"
    )

    c2.metric(
        "Test F1",
        f"{result['metrics']['F1']:.4f}"
    )

    c3.metric(
        "F1 Gap",
        f"{result['train_f1'] - result['metrics']['F1']:.4f}"
    )

    cm = confusion_matrix(
        result["y_test"],
        result["pred"]
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax
    )

    ax.set_title(
        "Random Forest — Confusion Matrix"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)

    st.markdown(
        "### Top 10 Feature Importances"
    )

    st.dataframe(
        result["importance"],
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    importance_table = result["importance"]

    ax.barh(
        importance_table["Feature"][::-1],
        importance_table["Importance"][::-1]
    )

    ax.set_xlabel(
        "Importance"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)


elif page == "XGBoost":

    st.subheader(
        "XGBoost — High Usage Classification"
    )

    (
        df_encoded,
        x_prepared,
        scale_columns,
        binary_columns
    ) = prepare_data(
        raw_df
    )

    with st.spinner(
        "Training XGBoost..."
    ):

        result = train_xgb(
            x_prepared
        )

    st.info(
        "Target: high_usage_flag | 0 = Normal Usage | 1 = High Usage"
    )

    st.dataframe(
        metric_table(
            result["metrics"]
        ),
        use_container_width=True
    )

    cm = confusion_matrix(
        result["y_test"],
        result["pred"]
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax
    )

    ax.set_title(
        "XGBoost — Confusion Matrix"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)

    st.code(
        result["report"]
    )


elif page == "Model Summary":

    st.subheader(
        "Four-Model Summary"
    )

    (
        df_encoded,
        x_prepared,
        scale_columns,
        binary_columns
    ) = prepare_data(
        raw_df
    )

    with st.spinner(
        "Preparing model results..."
    ):

        logistic_result = train_logistic(
            raw_df,
            df_encoded
        )

        ridge_result = train_ridge(
            raw_df
        )

        rf_result = train_rf(
            df_encoded
        )

        xgb_result = train_xgb(
            x_prepared
        )

    summary = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "Task": "Classification",
                "Target": "anomaly_flag",
                **logistic_result["metrics"]
            },
            {
                "Model": "Ridge Regression",
                "Task": "Regression",
                "Target": "next_hour_consumption_kwh",
                **ridge_result["metrics"]
            },
            {
                "Model": "Random Forest",
                "Task": "Classification",
                "Target": "anomaly_flag",
                **rf_result["metrics"]
            },
            {
                "Model": "XGBoost",
                "Task": "Classification",
                "Target": "high_usage_flag",
                **xgb_result["metrics"]
            }
        ]
    )

    st.dataframe(
        summary.round(4),
        use_container_width=True
    )

    st.warning(
        "Classification models do not share the same target: Logistic Regression and Random Forest use anomaly_flag, while XGBoost uses high_usage_flag. Ridge is a separate regression task, so MAE/RMSE/R² should not be directly compared with classification metrics."
    )
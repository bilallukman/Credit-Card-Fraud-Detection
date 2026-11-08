
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fraud Detection | AI Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "fraud_detection_model.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


pipeline = load_model()

model = pipeline["model"]
scaler = pipeline["scaler"]
threshold = pipeline["threshold"]
features = pipeline["features"]


# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}

.block-container {
    padding-top: 2rem;
}

[data-testid="stMetric"] {
    background: #151f32;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #26364f;
}

h1, h2, h3 {
    color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.title("🛡️ Credit Card Fraud Detection")

st.caption(
    "Machine Learning Powered Transaction Monitoring"
)

st.divider()

st.markdown("""
### Transaction Analysis

Upload a CSV file containing the 30 transaction features
required by our trained Random Forest model.

If your CSV contains the actual `Class` column, the dashboard
can also evaluate the model using the chronological test period.
""")


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        # ---------------- VALIDATE FILE ----------------
        missing_features = [
            col for col in features
            if col not in data.columns
        ]

        if missing_features:

            st.error(
                f"Missing required columns: {missing_features}"
            )

        elif data.empty:

            st.warning("The uploaded file is empty.")

        else:

            # Only model features go into the model.
            # Class is never used as a prediction input.
            X = data[features].copy()

            if not all(
                pd.api.types.is_numeric_dtype(X[col])
                for col in features
            ):

                st.error(
                    "All model features must contain numeric values."
                )

            elif X.isna().any().any():

                st.error(
                    "Your file contains missing values."
                )

            elif not np.isfinite(X.to_numpy()).all():

                st.error(
                    "Your file contains infinite values."
                )

            else:

                # ---------------- PREDICTIONS ----------------
                X_scaled = scaler.transform(X)

                probabilities = model.predict_proba(
                    X_scaled
                )[:, 1]

                predictions = (
                    probabilities >= threshold
                ).astype(int)

                results = data.copy()

                results["Fraud_Probability"] = probabilities

                results["Prediction"] = [
                    "Fraudulent" if p == 1 else "Legitimate"
                    for p in predictions
                ]

                
                # ---------------- HIGH-RISK TRANSACTION ALERTS ----------------
                st.divider()
                st.subheader(" High-Risk Transaction Alerts")

                st.write(
                    "Transactions predicted as fraudulent are listed "
                    "below, ranked by their fraud probability."
                )

                # Filter transactions flagged as fraudulent
                high_risk = results[
                    results["Prediction"] == "Fraudulent"
                ].copy()

                # Sort from highest to lowest fraud probability
                high_risk = high_risk.sort_values(
                    by="Fraud_Probability",
                    ascending=False
                ).reset_index(drop=True)

                if high_risk.empty:
                    st.success(
                        "No high-risk transactions detected in this upload."
                    )

                else:
                    # Summary metrics
                    alert_col1, alert_col2 = st.columns(2)

                    alert_col1.metric(
                        "High-Risk Transactions",
                        f"{len(high_risk):,}"
                    )

                    alert_col2.metric(
                        "Highest Fraud Probability",
                        f"{high_risk['Fraud_Probability'].max():.2%}"
                    )

                    st.warning(
                        f"⚠️ {len(high_risk):,} transactions require further review."
                    )

                    # Display important columns first
                    display_columns = [
                        col for col in [
                            "Time",
                            "Amount",
                            "Fraud_Probability",
                            "Prediction",
                            "Class"
                        ]
                        if col in high_risk.columns
                    ]

                    st.markdown("#### Suspicious Transaction List")

                    st.dataframe(
                        high_risk[display_columns].style.format(
                            {"Fraud_Probability": "{:.2%}"}
                        ),
                        use_container_width=True
                    )

                    # Download separate high-risk report
                    high_risk_csv = high_risk.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "⬇️ Download High-Risk Report",
                        data=high_risk_csv,
                        file_name="high_risk_transactions.csv",
                        mime="text/csv",
                        key="high_risk_download"
                    )


                
                # ---------------- INDIVIDUAL TRANSACTION INVESTIGATION ----------------
                st.divider()
                st.subheader("🔍 Individual Transaction Investigation")

                st.write(
                    "Select a transaction to inspect its fraud probability "
                    "and prediction."
                )

                selected_transaction = st.number_input(
                    "Enter Transaction Number",
                    min_value=1,
                    max_value=len(results),
                    value=1,
                    step=1
                )

                # Get the selected transaction
                selected_index = int(selected_transaction) - 1
                selected_row = results.iloc[selected_index]

                probability = float(
                    selected_row["Fraud_Probability"]
                )

                prediction = selected_row["Prediction"]

                # Display investigation results
                inv1, inv2, inv3 = st.columns(3)

                inv1.metric(
                    "Transaction Number",
                    f"#{selected_transaction}"
                )

                inv2.metric(
                    "Fraud Probability",
                    f"{probability:.2%}"
                )

                inv3.metric(
                    "Model Prediction",
                    prediction
                )

                # Show threshold comparison
                st.markdown("#### Fraud Risk Assessment")

                st.progress(probability)

                st.caption(
                    f"Fraud probability: {probability:.2%} | "
                    f"Decision threshold: {threshold:.0%}"
                )

                if probability >= threshold:
                    st.error(
                        "⚠️ HIGH RISK: This transaction has been "
                        "flagged as potentially fraudulent."
                    )
                else:
                    st.success(
                        "✅ LOW RISK: This transaction is classified "
                        "as legitimate by the model."
                    )

                # Display original transaction details
                with st.expander("View Transaction Feature Details"):
                    st.dataframe(
                        selected_row[features].to_frame(
                            name="Feature Value"
                        ),
                        use_container_width=True
                    )



                # ---------------- ANALYSIS OVERVIEW ----------------
                total = len(results)

                fraud_count = int(predictions.sum())

                legitimate_count = (
                    total - fraud_count
                )

                fraud_percentage = (
                    fraud_count / total
                ) * 100

                st.divider()

                st.subheader("Analysis Overview")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Total Transactions",
                    f"{total:,}"
                )

                col2.metric(
                    "Fraud Detected",
                    f"{fraud_count:,}"
                )

                col3.metric(
                    "Legitimate",
                    f"{legitimate_count:,}"
                )

                col4.metric(
                    "Fraud Rate",
                    f"{fraud_percentage:.2f}%"
                )


                # ==================================================
                # CHRONOLOGICAL HOLDOUT MODEL EVALUATION
                # ==================================================

                st.divider()

                st.subheader(
                    "Actual vs Predicted Model Evaluation"
                )

                if (
                    "Class" in data.columns
                    and "Time" in data.columns
                ):

                    # Remove duplicates and sort by transaction time
                    evaluation_data = (
                        data.drop_duplicates()
                        .sort_values("Time")
                        .reset_index(drop=True)
                    )

                    # Recreate the original chronological 80/20 split
                    split_point = int(
                        len(evaluation_data) * 0.80
                    )

                    test_data = (
                        evaluation_data
                        .iloc[split_point:]
                        .copy()
                    )

                    X_test = test_data[features]

                    y_test = pd.to_numeric(
                        test_data["Class"],
                        errors="coerce"
                    )

                    if (
                        y_test.isna().any()
                        or not y_test.isin([0, 1]).all()
                    ):

                        st.warning(
                            "Invalid Class labels. "
                            "Model evaluation is unavailable."
                        )

                    else:

                        # Use the saved scaler and trained model.
                        # Do not fit or retrain either one here.
                        X_test_scaled = scaler.transform(
                            X_test
                        )

                        test_probabilities = (
                            model.predict_proba(
                                X_test_scaled
                            )[:, 1]
                        )

                        test_predictions = (
                            test_probabilities >= threshold
                        ).astype(int)

                        y_test = y_test.astype(int)


                        # ---------------- EVALUATION METRICS ----------------
                        accuracy = accuracy_score(
                            y_test,
                            test_predictions
                        )

                        precision = precision_score(
                            y_test,
                            test_predictions,
                            zero_division=0
                        )

                        recall = recall_score(
                            y_test,
                            test_predictions,
                            zero_division=0
                        )

                        f1 = f1_score(
                            y_test,
                            test_predictions,
                            zero_division=0
                        )

                        avg_precision = (
                            average_precision_score(
                                y_test,
                                test_probabilities
                            )
                        )


                        # ---------------- CONFUSION MATRIX ----------------
                        cm = confusion_matrix(
                            y_test,
                            test_predictions,
                            labels=[0, 1]
                        )

                        tn, fp, fn, tp = cm.ravel()


                        st.success(
                            "Chronological holdout evaluation "
                            "completed successfully."
                        )

                        st.caption(
                            f"Evaluation uses the final 20% of "
                            f"{len(evaluation_data):,} unique transactions. "
                            f"Test transactions: {len(test_data):,}."
                        )


                        # ---------------- METRICS DISPLAY ----------------
                        st.markdown(
                            "#### Holdout Evaluation Metrics"
                        )

                        m1, m2, m3, m4, m5 = st.columns(5)

                        m1.metric(
                            "Accuracy",
                            f"{accuracy:.2%}"
                        )

                        m2.metric(
                            "Precision",
                            f"{precision:.2%}"
                        )

                        m3.metric(
                            "Recall",
                            f"{recall:.2%}"
                        )

                        m4.metric(
                            "F1-Score",
                            f"{f1:.2%}"
                        )

                        m5.metric(
                            "Average Precision",
                            f"{avg_precision:.2%}"
                        )


                        # ---------------- CONFUSION MATRIX DISPLAY ----------------
                        st.markdown("#### Confusion Matrix")

                        matrix_data = pd.DataFrame(
                            cm,
                            index=[
                                "Actual Legitimate",
                                "Actual Fraud"
                            ],
                            columns=[
                                "Predicted Legitimate",
                                "Predicted Fraud"
                            ]
                        )

                        st.dataframe(
                            matrix_data,
                            use_container_width=True
                        )


                        # ---------------- PREDICTION BREAKDOWN ----------------
                        st.markdown(
                            "#### Prediction Breakdown"
                        )

                        c1, c2, c3, c4 = st.columns(4)

                        c1.metric(
                            "True Negatives",
                            f"{tn:,}"
                        )

                        c2.metric(
                            "False Positives",
                            f"{fp:,}"
                        )

                        c3.metric(
                            "False Negatives",
                            f"{fn:,}"
                        )

                        c4.metric(
                            "True Positives",
                            f"{tp:,}"
                        )

                        st.caption(
                            "Precision measures how many flagged "
                            "transactions were actually fraud. "
                            "Recall measures how many actual fraud "
                            "transactions were detected."
                        )

                else:

                    st.info(
                        "For chronological evaluation, upload a "
                        "labelled dataset containing both Time "
                        "and Class columns."
                    )


                # ---------------- RESULTS TABLE ----------------
                st.divider()

                st.subheader(
                    "Transaction Prediction Results"
                )

                st.dataframe(
                    results,
                    use_container_width=True
                )


               
                # ---------------- TRANSACTION DISTRIBUTION ----------------
                st.divider()
                st.subheader("Transaction Distribution")

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:

                    st.markdown("#### Predicted Transaction Classes")

                    class_data = pd.DataFrame({
                        "Transaction Type": [
                            "Legitimate",
                            "Fraudulent"
                        ],
                        "Percentage": [
                            legitimate_count / total * 100,
                            fraud_count / total * 100
                        ]
                    })

                    st.bar_chart(
                        class_data.set_index("Transaction Type"),
                        horizontal=True
                    )

                    st.caption(
                        "Percentage of transactions classified into each category."
                    )

                with chart_col2:

                    st.markdown("#### Fraud Probability Distribution")

                    probability_bins = [
                        0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.01
                    ]

                    probability_labels = [
                        "0–1%",
                        "1–5%",
                        "5–10%",
                        "10–20%",
                        "20–50%",
                        "50–100%"
                    ]

                    results["Probability_Range"] = pd.cut(
                        results["Fraud_Probability"],
                        bins=probability_bins,
                        labels=probability_labels,
                        include_lowest=True,
                        right=False
                    )

                    probability_data = (
                        results["Probability_Range"]
                        .value_counts(sort=False)
                        .rename_axis("Probability Range")
                        .reset_index(name="Transactions")
                    )

                    st.bar_chart(
                        probability_data.set_index("Probability Range")
                    )

                    st.caption(
                        "Number of transactions falling into each predicted fraud probability range."
                    )



                # ---------------- DOWNLOAD RESULTS ----------------
                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Prediction Results",
                    data=csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )

                st.success(
                    "Transaction analysis completed successfully."
                )


    except Exception as e:

        st.error(
            f"An error occurred while processing the file: {e}"
        )


# ---------------- FOOTER ----------------
st.divider()

st.caption(
    f"Credit Card Fraud Detection | Random Forest | "
    f"Threshold: {threshold:.2f}"
)

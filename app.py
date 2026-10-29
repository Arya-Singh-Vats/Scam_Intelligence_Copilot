import json
from pathlib import Path

import pandas as pd
import streamlit as st

from detector import analyze_message
from transformer_detector import analyze_with_distilbert
from llm_explainer import explain_message
from evaluation import get_evaluation_results
from threat_analyzer import analyze_threat_indicators


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Scam Intelligence Copilot",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
COMPARISON_PATH = BASE_DIR / "models" / "model_comparison.json"


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def normalize_prediction(prediction):
    """Normalize predictions from the two classifiers."""
    value = str(prediction).strip().lower()

    if value in {"spam", "spam-like", "spam_like"}:
        return "spam"

    if value in {"ham", "legitimate", "legitimate (ham)", "not spam"}:
        return "ham"

    return value


def get_risk_category(spam_probability):
    """
    Convert a model score into a preliminary display category.
    These thresholds are heuristic, not calibrated fraud probabilities.
    """
    if spam_probability >= 0.75:
        return "High"
    if spam_probability >= 0.40:
        return "Medium"
    return "Low"


def load_comparison_metrics():
    """Load saved evaluation metrics for both classifiers."""
    if not COMPARISON_PATH.exists():
        return None

    try:
        with COMPARISON_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def render_threat_indicators(indicator_result):
    """Render the actual schema returned by threat_analyzer.py."""
    st.subheader("🚩 Threat Indicators")

    if not indicator_result:
        st.write("No threat indicators were returned.")
        return

    # Current threat_analyzer.py returns a dictionary containing
    # indicators, indicator_count, urls_found, and status.
    if isinstance(indicator_result, dict):
        items = indicator_result.get("indicators", [])
        status = indicator_result.get(
            "status",
            "Potential warning signs detected" if items
            else "No configured warning patterns detected",
        )
        count = indicator_result.get("indicator_count", len(items))
        urls_found = indicator_result.get("urls_found", [])
    elif isinstance(indicator_result, list):
        items = indicator_result
        status = (
            "Potential warning signs detected"
            if items else "No configured warning patterns detected"
        )
        count = len(items)
        urls_found = []
    else:
        st.write(indicator_result)
        return

    st.write(f"**Status:** {status}")
    st.write(f"**Warning indicators found:** {count}")

    if not items:
        st.caption(
            "No configured warning patterns were detected. "
            "This does not guarantee that the message is safe."
        )
    else:
        for item in items:
            if isinstance(item, dict):
                # IMPORTANT: threat_analyzer.py uses 'category' and 'details',
                # not 'name' and 'description'.
                category = item.get(
                    "category",
                    item.get("name", item.get("indicator", "Warning")),
                )
                severity = item.get(
                    "severity",
                    item.get("risk", "Unspecified"),
                )
                details = item.get(
                    "details",
                    item.get("description", ""),
                )

                st.markdown(f"- **{category}** — {severity}")

                if details:
                    st.caption(details)

                evidence = item.get("evidence", [])
                if evidence:
                    st.caption(
                        "Evidence: "
                        + ", ".join(str(value) for value in evidence)
                    )
            else:
                st.markdown(f"- {item}")

    if urls_found:
        st.caption(f"URL(s) detected: {len(urls_found)}")


def render_baseline_evaluation():
    """Render the existing Baseline ML evaluation dashboard."""
    st.divider()
    st.header("📊 Baseline ML Evaluation")

    try:
        evaluation_results = get_evaluation_results()

        if not evaluation_results:
            st.info("No baseline evaluation results are available.")
            return

        # Accommodate common metric key names.
        metric_aliases = {
            "Accuracy": ["accuracy", "Accuracy"],
            "Precision": ["precision", "spam_precision", "Precision"],
            "Recall": ["recall", "spam_recall", "Recall"],
            "F1-score": ["f1", "f1_score", "spam_f1", "F1-score"],
        }

        metrics_found = {}
        for label, aliases in metric_aliases.items():
            for alias in aliases:
                if alias in evaluation_results:
                    metrics_found[label] = evaluation_results[alias]
                    break

        if metrics_found:
            columns = st.columns(len(metrics_found))
            for column, (label, value) in zip(columns, metrics_found.items()):
                with column:
                    # Evaluation metrics are expected as fractions from 0 to 1.
                    st.metric(label, f"{float(value) * 100:.2f}%")
        else:
            st.json(evaluation_results)

        if "confusion_matrix" in evaluation_results:
            st.markdown("#### Confusion Matrix")
            st.write(evaluation_results["confusion_matrix"])

    except Exception as exc:
        st.warning(
            "The baseline evaluation dashboard could not load. "
            "Your message-analysis feature may still work."
        )
        st.caption(str(exc))


def render_model_comparison():
    """Render the saved comparison between Baseline ML and DistilBERT."""
    st.divider()
    st.header("📈 Model Comparison Dashboard")

    comparison = load_comparison_metrics()

    if comparison is None:
        st.warning(
            "The model comparison file could not be loaded. "
            "Check models/model_comparison.json."
        )
        return

    baseline = comparison.get("baseline", {})
    distilbert = comparison.get("distilbert", {})

    st.caption(
        f"Recorded test samples: {comparison.get('test_samples', 'Unknown')}"
    )

    metric_definitions = [
        ("Accuracy", "accuracy"),
        ("Spam Precision", "spam_precision"),
        ("Spam Recall", "spam_recall"),
        ("Spam F1-score", "spam_f1"),
    ]

    rows = []
    for label, key in metric_definitions:
        if key in baseline and key in distilbert:
            rows.append({
                "Metric": label,
                "Baseline ML (%)": round(float(baseline[key]) * 100, 2),
                "DistilBERT (%)": round(float(distilbert[key]) * 100, 2),
            })

    if not rows:
        st.warning(
            "The comparison file is missing one or more expected metric keys."
        )
        return

    comparison_df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baseline ML")
        if "accuracy" in baseline:
            st.metric("Accuracy", f"{float(baseline['accuracy']) * 100:.2f}%")
        if "spam_precision" in baseline:
            st.metric(
                "Spam Precision",
                f"{float(baseline['spam_precision']) * 100:.2f}%",
            )
        if "spam_recall" in baseline:
            st.metric(
                "Spam Recall",
                f"{float(baseline['spam_recall']) * 100:.2f}%",
            )
        if "spam_f1" in baseline:
            st.metric(
                "Spam F1-score",
                f"{float(baseline['spam_f1']) * 100:.2f}%",
            )

    with col2:
        st.subheader("DistilBERT")
        if "accuracy" in distilbert:
            st.metric(
                "Accuracy",
                f"{float(distilbert['accuracy']) * 100:.2f}%",
            )
        if "spam_precision" in distilbert:
            st.metric(
                "Spam Precision",
                f"{float(distilbert['spam_precision']) * 100:.2f}%",
            )
        if "spam_recall" in distilbert:
            st.metric(
                "Spam Recall",
                f"{float(distilbert['spam_recall']) * 100:.2f}%",
            )
        if "spam_f1" in distilbert:
            st.metric(
                "Spam F1-score",
                f"{float(distilbert['spam_f1']) * 100:.2f}%",
            )

    st.subheader("Metric-by-Metric Comparison")
    st.dataframe(
        comparison_df,
        hide_index=True,
        use_container_width=True,
    )

    # Both models are compared on the same percentage scale (0-100).
    chart_df = comparison_df.set_index("Metric")
    st.bar_chart(chart_df, y=["Baseline ML (%)", "DistilBERT (%)"])

    st.subheader("Interpretation")

    if (
        "accuracy" in baseline and "accuracy" in distilbert
        and "spam_precision" in baseline and "spam_precision" in distilbert
        and "spam_recall" in baseline and "spam_recall" in distilbert
        and "spam_f1" in baseline and "spam_f1" in distilbert
    ):
        st.markdown(
            f"- **Baseline ML accuracy:** {float(baseline['accuracy']) * 100:.2f}%\n"
            f"- **DistilBERT accuracy:** {float(distilbert['accuracy']) * 100:.2f}%\n"
            f"- **Baseline ML spam recall:** {float(baseline['spam_recall']) * 100:.2f}%\n"
            f"- **DistilBERT spam precision:** {float(distilbert['spam_precision']) * 100:.2f}%"
        )

        st.write(
            "In these recorded results, Baseline ML has higher recall and "
            "F1-score, while DistilBERT has higher spam precision. Accuracy, "
            "precision, recall, and F1-score measure different aspects of "
            "performance."
        )

    st.warning(
        "These are recorded experimental results, not a guarantee of "
        "real-world performance. Before reporting them as a fair comparison, "
        "verify that both models were trained without test-set leakage and "
        "evaluated on exactly the same held-out messages."
    )


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_message" not in st.session_state:
    st.session_state.analysis_message = ""

if "analysis_model" not in st.session_state:
    st.session_state.analysis_model = "Baseline ML"

if "ai_explanation" not in st.session_state:
    st.session_state.ai_explanation = None

if "threat_indicators" not in st.session_state:
    st.session_state.threat_indicators = None


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🛡️ Scam Intelligence Copilot")
st.markdown(
    "Machine learning, Transformer-based detection, "
    "and local AI cybersecurity explanations"
)
st.caption(
    "Educational cybersecurity prototype. Predictions are estimates, "
    "not proof of fraud."
)
st.divider()


# --------------------------------------------------
# MESSAGE ANALYSIS
# --------------------------------------------------

st.header("🔍 Analyze a Message")

selected_model = st.selectbox(
    "Model used",
    ["Baseline ML", "DistilBERT (Transformer)"],
    key="selected_model",
)

message = st.text_area(
    "Enter a suspicious message",
    height=150,
    placeholder=(
        "Example: URGENT: Your bank account will be blocked. "
        "Click this link to verify your password."
    ),
)

if st.button("Analyze Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message to analyze.")
    else:
        try:
            with st.spinner("Analyzing message..."):
                if selected_model == "Baseline ML":
                    raw_result = analyze_message(message)
                else:
                    raw_result = analyze_with_distilbert(message)

                prediction = normalize_prediction(raw_result["prediction"])
                spam_probability = float(raw_result["spam_probability"])

                if not 0.0 <= spam_probability <= 1.0:
                    raise ValueError(
                        "The model spam score must be between 0 and 1."
                    )

                if prediction not in {"spam", "ham"}:
                    raise ValueError(
                        "The selected model returned an unexpected "
                        f"prediction: {prediction}"
                    )

                risk = get_risk_category(spam_probability)

                st.session_state.analysis_result = {
                    "prediction": prediction,
                    "spam_probability": spam_probability,
                    "risk": risk,
                }
                st.session_state.analysis_message = message
                st.session_state.analysis_model = selected_model

                st.session_state.threat_indicators = (
                    analyze_threat_indicators(message)
                )

                st.session_state.ai_explanation = explain_message(
                    message=message,
                    prediction=prediction,
                    spam_probability=spam_probability,
                    risk=risk,
                )

        except Exception as exc:
            st.error(
                "Analysis failed. Check the terminal for details and "
                "verify that your models and Ollama are available."
            )
            st.exception(exc)


# --------------------------------------------------
# DISPLAY ANALYSIS RESULTS
# --------------------------------------------------

if st.session_state.analysis_result is not None:
    result = st.session_state.analysis_result

    st.subheader("Analysis Results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model used**")
        st.write(st.session_state.analysis_model)

        st.markdown("**Model prediction**")
        if result["prediction"] == "spam":
            st.error("Spam-like")
        else:
            st.success("Legitimate (ham)")

    with col2:
        st.metric(
            "Estimated spam probability",
            f"{result['spam_probability'] * 100:.1f}%",
        )
        st.markdown("### Preliminary risk category")

        if result["risk"] == "High":
            st.error("High")
        elif result["risk"] == "Medium":
            st.warning("Medium")
        else:
            st.success("Low")

    st.info(
        "This system is an educational prototype. Its spam probability "
        "is a model score, not proof of fraud. A low score does not "
        "guarantee that a message is safe."
    )

    st.divider()
    render_threat_indicators(st.session_state.threat_indicators)

    st.divider()
    st.subheader("🤖 AI-Powered Scam Analysis")

    explanation = st.session_state.ai_explanation
    if isinstance(explanation, str):
        st.markdown(explanation)
    elif isinstance(explanation, dict):
        # Support dict-style output if the explainer is changed later.
        for key, value in explanation.items():
            st.markdown(f"### {key.replace('_', ' ').title()}")
            if isinstance(value, list):
                for item in value:
                    st.markdown(f"- {item}")
            elif isinstance(value, dict):
                st.json(value)
            else:
                st.markdown(str(value))
    elif explanation is None:
        st.warning("No AI explanation is available yet.")
    else:
        st.write(explanation)

    st.caption(
        "Never share passwords or OTPs in response to unexpected messages. "
        "Verify requests through official channels."
    )


# --------------------------------------------------
# EVALUATION DASHBOARDS
# --------------------------------------------------

render_baseline_evaluation()
render_model_comparison()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "Scam Intelligence Copilot | Educational prototype | "
    "Always verify suspicious requests through official channels."
)

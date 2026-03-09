from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
from typing import Any, Dict, List
from pathlib import Path
import json
import pandas as pd
import streamlit as st

# Import the rules engine from the clean folder structure
from rules.risk_rules import assess_with_rules
from ml.text_classifier import predict_risk_from_text

# Single source of truth for versioning (used in payload + export)
TOOL_VERSION = "thesis-prototype-2026"

# Project paths (robust against different working directories)
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "ml" / "model.joblib"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "training_data.csv"
METRICS_PATH = PROJECT_ROOT / "ml" / "metrics.json"

# Single source of truth for widget defaults (used for init + reset)
FORM_DEFAULTS = {
    # Section 1) Use case basics
    "use_case_name": "",
    "use_case_description": "",
    "primary_goal": "Decision support",
    "domain_context": "HR / recruitment",
    "intended_users": ["Internal employees"],
    "affected_groups": [],

    # Section 2) Decision & impact
    "decision_type": "Decision support (human approves)",
    "decision_criticality": "Medium (access to services, moderate effects)",
    "human_in_loop": "Yes, always",
    "appeal_possible": "Unknown",
    "scale": "Small scale pilot",

    # Section 3) Data & privacy
    "personal_data": "Unknown",
    "sensitive_data_types": [],
    "data_sources": ["Internal company records"],
    "retention_defined": "Unknown",
    "cross_border_sharing": "Unknown",

    # Section 4) System & governance signals
    "ai_approach_types": ["Classical ML"],
    "model_transparency": "Unknown",
    "training_data_confidence": "Unknown",
    "monitoring_planned": "Unknown",
    "misuse_concern": "Unknown",

    # Section 5) Responsible AI & documentation signals
    "bias_risk": "Unknown",
    "protected_groups_affected": "Unknown",
    "accountable_owner_defined": "Unknown",
    "documentation_available": ["None yet"],
    "known_frameworks": ["None / unknown"],
}

# Initialize form state once (so widgets get defaults from session_state)
for k, v in FORM_DEFAULTS.items():
    st.session_state.setdefault(k, v)


# Reset helper (runs as a callback)
def reset_form_state() -> None:
    """
    Reset must happen via callback (on_click), otherwise Streamlit may throw:
    'st.session_state.<key> cannot be modified after the widget is instantiated'

    Defaults are controlled ONLY via session_state to avoid Streamlit warnings about
    "widget created with a default value but also set via Session State API".
    """
    # Clear generated results
    st.session_state.pop("submitted_payload", None)

    # Restore widget defaults
    for k, v in FORM_DEFAULTS.items():
        st.session_state[k] = v


# Helper for hybrid aggregation
def _level_order(level: str) -> int:
    order = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }
    return order.get(level, -1)


def _normalize_level(label: str) -> str:
    """Normalize labels to Low/Medium/High where possible (ML returns lower-case)."""
    if not isinstance(label, str):
        return ""
    s = label.strip()
    lower = s.lower()
    if lower in {"low", "medium", "high"}:
        return lower.capitalize()
    return s


def _class_probs_to_dict(obj) -> dict:
    """
    text_classifier.MLResult.class_probabilities is currently List[Tuple[str, float]].
    This normalizes it into a dict for export + UI.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {str(k): float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        out = {}
        for item in obj:
            try:
                k, v = item
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    return {}


# Page setup
st.set_page_config(
    page_title="Responsible AI Risk & Compliance Support Tool (MVP)",
    layout="wide",
)

st.title("Responsible AI Risk & Compliance Support Tool (MVP)")
st.caption("Early-phase decision-support prototype (not legal advice / not a compliance decision).")


# Small UI helpers
def yes_no_unknown(label: str, key: str):
    return st.selectbox(label, ["Unknown", "Yes", "No"], key=key)


def low_med_high_unknown(label: str, key: str):
    return st.selectbox(label, ["Unknown", "Low", "Medium", "High"], key=key)


def required_text_area(label: str, key: str, min_chars: int = 50):
    val = st.text_area(label, key=key, height=140, placeholder="Write 3–10 sentences...")
    if val and len(val.strip()) < min_chars:
        st.warning(f"Please add a bit more detail (min {min_chars} characters).")
    return val


# Payload building + validation
def build_payload() -> Dict[str, Any]:
    # Build a structured JSON-like object from Streamlit session state
    return {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_version": TOOL_VERSION,
        },
        "use_case_basics": {
            "use_case_name": st.session_state.get("use_case_name", "").strip(),
            "use_case_description": st.session_state.get("use_case_description", "").strip(),
            "primary_goal": st.session_state.get("primary_goal"),
            "domain_context": st.session_state.get("domain_context"),
            "intended_users": st.session_state.get("intended_users", []),
            "affected_groups": st.session_state.get("affected_groups", []),
        },
        "decision_impact": {
            "decision_type": st.session_state.get("decision_type"),
            "decision_criticality": st.session_state.get("decision_criticality"),
            "human_in_loop": st.session_state.get("human_in_loop"),
            "appeal_possible": st.session_state.get("appeal_possible"),
            "scale": st.session_state.get("scale"),
        },
        "data_privacy": {
            "personal_data": st.session_state.get("personal_data"),
            "sensitive_data_types": st.session_state.get("sensitive_data_types", []),
            "data_sources": st.session_state.get("data_sources", []),
            "retention_defined": st.session_state.get("retention_defined"),
            "cross_border_sharing": st.session_state.get("cross_border_sharing"),
        },
        "system_model": {
            "ai_approach_types": st.session_state.get("ai_approach_types", []),
            "model_transparency": st.session_state.get("model_transparency"),
            "training_data_confidence": st.session_state.get("training_data_confidence"),
            "monitoring_planned": st.session_state.get("monitoring_planned"),
            "misuse_concern": st.session_state.get("misuse_concern"),
        },
        "governance": {
            "bias_risk": st.session_state.get("bias_risk"),
            "protected_groups_affected": st.session_state.get("protected_groups_affected"),
            "accountable_owner_defined": st.session_state.get("accountable_owner_defined"),
            "documentation_available": st.session_state.get("documentation_available", []),
            "known_frameworks": st.session_state.get("known_frameworks", []),
        },
    }


def validate_payload(payload: Dict[str, Any]) -> List[str]:
    # Minimal validation so the assessment output is meaningful
    errors: List[str] = []
    basics = payload["use_case_basics"]

    if not basics["use_case_name"]:
        errors.append("Use case name is required.")
    desc = basics["use_case_description"]
    if not desc or len(desc) < 50:
        errors.append("Use case description is required (min 50 characters).")
    if not basics["domain_context"]:
        errors.append("Domain / context is required.")

    return errors


def get_model_status() -> dict:
    """
    Returns a small status dict for UI display.
    This does NOT train or modify the model.
    """
    status = {
        "loaded": False,
        "train_rows": None,
        "last_trained": None,
        "message": None,
    }

    # Training data size (if file exists)
    try:
        if TRAIN_DATA_PATH.exists():
            df = pd.read_csv(TRAIN_DATA_PATH)
            status["train_rows"] = int(len(df))
        else:
            status["message"] = "Training data not found (data/training_data.csv)."
    except Exception as e:
        status["message"] = f"Could not read training data: {e}"

    # Model file status + last modified timestamp
    try:
        if MODEL_PATH.exists():
            status["loaded"] = True
            ts = MODEL_PATH.stat().st_mtime  # last modified time (epoch)
            status["last_trained"] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        else:
            status["message"] = (status["message"] or "") + " Model not found (ml/model.joblib)."
    except Exception as e:
        status["message"] = (status["message"] or "") + f" Could not read model file metadata: {e}"

    return status


# Model status (sidebar)
with st.sidebar:
    st.subheader("Model status")

    model_status = get_model_status()

    if model_status["loaded"]:
        st.write("✔ Model loaded")
    else:
        st.write("✖ Model not loaded")

    if model_status["train_rows"] is not None:
        st.write(f"Training data size: {model_status['train_rows']} synthetic scenarios")
    else:
        st.write("Training data size: Unknown")

    if model_status["last_trained"] is not None:
        st.write(f"Last trained: {model_status['last_trained']}")
    else:
        st.write("Last trained: Unknown")


    st.divider()
    st.subheader("Model evaluation")

    if METRICS_PATH.exists():
        with st.expander("Show evaluation metrics"):
            try:
                metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

                acc = metrics_data.get("metrics", {}).get("accuracy", None)
                macro_f1 = metrics_data.get("metrics", {}).get("macro_f1", None)

                if acc is not None:
                    st.write(f"**Accuracy:** {acc:.3f}")
                else:
                    st.write("**Accuracy:** (not found)")

                if macro_f1 is not None:
                    st.write(f"**Macro F1:** {macro_f1:.3f}")
                else:
                    st.write("**Macro F1:** (not found)")

                # Confusion matrix
                cm_block = metrics_data.get("confusion_matrix", {})
                labels = cm_block.get("labels", [])
                matrix = cm_block.get("matrix", [])

                if labels and matrix:
                    st.write("**Confusion matrix:**")
                    cm_df = pd.DataFrame(matrix, index=labels, columns=labels)
                    st.dataframe(cm_df, width="stretch")
                else:
                    st.write("Confusion matrix not found in metrics.json")

                # Optional: show full classification report
                report = metrics_data.get("classification_report", None)
                if report:
                    with st.expander("Show full classification report"):
                        st.json(report)

            except Exception as e:
                st.warning("Could not load metrics.json")
                st.caption(str(e))
    else:
        st.caption("No evaluation metrics found yet. Run: python -m ml.text_classifier train-eval")


    st.divider()
    st.caption("Model lifecycle transparency: The classifier is trained externally and loaded at runtime. Evaluation is based on synthetic data.")


# Layout: form + output preview
st.subheader("1) Describe the AI use case")

st.text_input("Use case name *", key="use_case_name", placeholder="e.g., AI-assisted CV screening for junior roles")

required_text_area("Use case description (free text) *", key="use_case_description", min_chars=50)

st.selectbox(
    "Primary goal",
    [
        "Efficiency / automation",
        "Decision support",
        "Personalization / recommendations",
        "Risk detection (fraud, security)",
        "Compliance monitoring",
        "Other",
    ],
    key="primary_goal",
)

st.selectbox(
    "Domain / context *",
    [
        "HR / recruitment",
        "Finance / credit / insurance",
        "Healthcare",
        "Education",
        "Public sector",
        "Security / law enforcement",
        "Marketing / customer analytics",
        "Industrial / IoT",
        "Internal operations",
        "Other",
    ],
    key="domain_context",
)

st.multiselect(
    "Intended users",
    ["Internal employees", "Customers", "Citizens / public", "Business partners", "Other"],
    key="intended_users",
)

st.multiselect(
    "Who is affected by the outcome?",
    [
        "Customers/users",
        "Employees/job applicants",
        "Patients/students",
        "General public",
        "Children/minors",
        "Vulnerable groups (general flag)",
    ],
    key="affected_groups",
)

st.divider()
st.subheader("2) Decision & impact characteristics")

st.selectbox(
    "Decision type",
    [
        "Recommendation only (human decides)",
        "Decision support (human approves)",
        "Automated decision (no human approval)",
        "Automated action (system triggers actions)",
    ],
    key="decision_type",
)

st.selectbox(
    "Decision criticality / impact",
    [
        "Low (convenience, minor effects)",
        "Medium (access to services, moderate effects)",
        "High (employment, finance, health, rights)",
    ],
    key="decision_criticality",
)

st.selectbox("Human in the loop?", ["Yes, always", "Sometimes / exception-based", "No"], key="human_in_loop")

yes_no_unknown("Right to contest / appeal planned?", key="appeal_possible")

st.selectbox(
    "Frequency / scale",
    ["Small scale pilot", "Department-level", "Organization-wide", "Public-facing / large scale"],
    key="scale",
)

st.divider()
st.subheader("3) Data & privacy (signals)")

yes_no_unknown("Personal data processed?", key="personal_data")

st.multiselect(
    "Special category (sensitive) data? (select all that apply)",
    ["Health", "Biometric", "Ethnic origin", "Political opinions", "Union membership", "Sexual orientation"],
    key="sensitive_data_types",
)

st.multiselect(
    "Data source(s)",
    [
        "Collected directly from individuals",
        "Internal company records",
        "Public sources",
        "Third-party vendor",
        "Web scraped",
        "Synthetic / generated",
    ],
    key="data_sources",
)

yes_no_unknown("Data retention defined?", key="retention_defined")
yes_no_unknown("Cross-border / external sharing expected?", key="cross_border_sharing")

st.divider()
st.subheader("4) System & governance signals")

st.multiselect(
    "AI approach type(s)",
    ["Rule-based", "Classical ML", "Deep learning", "Generative AI / LLM", "Computer vision", "NLP", "Unknown/not decided"],
    key="ai_approach_types",
)

st.selectbox(
    "Model transparency",
    ["High (interpretable)", "Medium", "Low (black-box)", "Unknown"],
    key="model_transparency",
)

st.selectbox(
    "Training data quality confidence",
    ["High", "Medium", "Low", "Unknown"],
    key="training_data_confidence",
)

yes_no_unknown("Monitoring planned post-deployment?", key="monitoring_planned")
low_med_high_unknown("Security / misuse concern", key="misuse_concern")

st.divider()
st.subheader("5) Responsible AI & documentation signals")

low_med_high_unknown("Potential bias risk", key="bias_risk")
yes_no_unknown("Protected groups potentially affected?", key="protected_groups_affected")
yes_no_unknown("Accountable owner defined?", key="accountable_owner_defined")

st.multiselect(
    "Documentation available",
    ["Requirements document", "Data sources documented", "Model documentation", "Risk assessment exists", "None yet"],
    key="documentation_available",
)

st.multiselect(
    "Known applicable frameworks (signals)",
    ["GDPR", "EU AI Act relevance suspected", "NIST AI RMF used internally", "ISO/IEC 42001-style governance", "None / unknown"],
    key="known_frameworks",
)

st.divider()
st.subheader("Preview & submit")

payload = build_payload()
errors = validate_payload(payload)

if errors:
    st.error("Please fix these before submitting:\n- " + "\n- ".join(errors))

col_a, col_b = st.columns(2)

# Store submitted payload in session state so the page can rerender without losing results
with col_a:
    if st.button("Generate assessment", width="stretch", disabled=bool(errors)):
        st.session_state["submitted_payload"] = payload

# Reset: callback-based reset so widgets truly reset without session reload
with col_b:
    st.button(
        "Reset form",
        width="stretch",
        on_click=reset_form_state,
    )

# Results section (rules-based)
if "submitted_payload" in st.session_state:
    st.success("Assessment generated.")

    # Show the captured input JSON for transparency/debugging
    with st.expander("Show captured input JSON"):
        st.json(st.session_state["submitted_payload"])

    # Run the rules engine from rules/risk_rules.py
    result = assess_with_rules(st.session_state["submitted_payload"])

    st.subheader("Assessment results (rules-based)")

    # Headline with visual emphasis
    if result.assessment_readiness == "Insufficient information":
        st.error(f"Overall attention level: {result.overall_attention}")

    elif result.assessment_readiness == "Provisional":
        st.warning(f"Overall attention level: {result.overall_attention}")

    else:
        # readiness = Ready
        if result.overall_attention == "High":
            st.warning(f"Overall attention level: {result.overall_attention}")
        else:
            # Low or Medium
            st.metric("Overall attention level", result.overall_attention)

    st.subheader("Assessment readiness / information quality")
    st.write(f"**Readiness:** {result.assessment_readiness}")
    st.write(f"**Unknown fields count:** {result.unknown_fields_count}")

    if result.critical_unknowns:
        st.warning("Critical information is missing. The assessment headline is not final.")
        st.write("**Critical unknowns:**")
        for x in result.critical_unknowns:
            st.write(f"- {x}")

    # Show what attention would be if complete
    st.caption(f"Attention level (if information were complete): {result.attention_level_if_complete}")

    st.subheader("Domain risk profile")
    profile_rows = []
    for domain, assessment in result.domain_profile.items():
        profile_rows.append(
            {
                "Domain": domain,
                "Level": assessment.level,
                "Top reasons": " | ".join(assessment.reasons[:3]),
            }
        )

    st.dataframe(profile_rows, width="stretch")

    with st.expander("Show full domain reasoning"):
        for domain, assessment in result.domain_profile.items():
            st.write(f"#### {domain} — {assessment.level}")
            for r in assessment.reasons:
                st.write(f"- {r}")

    st.write("**Risk categories flagged:**")
    st.write(result.risk_categories)

    st.write("**Compliance flags / signals:**")
    st.write(result.compliance_flags)

    st.write("**Why this was flagged (rule explanations):**")
    for ex in result.explanations:
        st.write(f"- {ex}")

    st.subheader("Actionable governance checklist")

    if result.recommended_actions:
        for a in result.recommended_actions:
            st.write(f"**[{a.priority}]** {a.action}")
            st.caption(a.rationale)
    else:
        st.write("No action items generated.")

    st.caption(
        "Disclaimer: This output is decision-support only and does not constitute legal advice "
        "or a compliance decision."
    )

    # ML: supporting signal based on free-text description
    use_case_text = (
        st.session_state["submitted_payload"]
        .get("use_case_basics", {})
        .get("use_case_description", "")
    )

    ml_result = None
    ml_export = None

    # Defensive check for robustness
    # (Future-proof if description becomes optional or external payloads are processed)
    if not use_case_text.strip():
        st.info("No free-text description provided, so ML classifier was skipped.")
    else:
        try:
            ml_result = predict_risk_from_text(use_case_text)

            st.divider()
            st.subheader("ML text classifier (supporting signal)")
            st.write(f"**Predicted label:** {ml_result.predicted_label}")
            st.write(f"**Confidence:** {ml_result.confidence:.2f}")

            probs_dict = _class_probs_to_dict(ml_result.class_probabilities)

            with st.expander("Show class probabilities"):
                st.json(probs_dict if probs_dict else ml_result.class_probabilities)

            ml_export = {
                "predicted_label": ml_result.predicted_label,
                "confidence": float(ml_result.confidence),
                "class_probabilities": probs_dict,
            }

        except FileNotFoundError as e:
            st.warning("ML model not trained yet. Train it with: python -m ml.text_classifier train")
            st.caption(str(e))
        except Exception as e:
            st.warning("ML classifier failed unexpectedly.")
            st.caption(str(e))

    # Final hybrid attention
    st.divider()
    st.subheader("Final hybrid attention decision")

    confidence_threshold = 0.75

    # Start from rules attention (when complete)
    rules_attention = _normalize_level(result.attention_level_if_complete)
    final_attention = rules_attention
    aggregation_explanation = "Rules-based assessment used as primary signal."

    if result.assessment_readiness != "Ready":
        final_attention = result.overall_attention
        aggregation_explanation = (
            "Final attention reflects information-quality status. "
            "No ML escalation applied when readiness is not Ready."
        )

        # Always show an output, even when readiness is not Ready
        if final_attention == "Insufficient information":
            st.error(f"Final attention: {final_attention}")
        elif final_attention == "Provisional":
            st.warning(f"Final attention: {final_attention}")
        elif final_attention == "High":
            st.warning(f"Final attention: {final_attention}")
        elif final_attention == "Medium":
            st.info(f"Final attention: {final_attention}")
        else:
            st.success(f"Final attention: {final_attention}")

        st.caption(aggregation_explanation)

    else:
        # readiness = Ready: allow ML escalation-only
        if ml_result is not None:
            ml_level = _normalize_level(ml_result.predicted_label)
            ml_conf = float(ml_result.confidence)

            if _level_order(ml_level) > _level_order(rules_attention) and ml_conf >= confidence_threshold:
                final_attention = ml_level
                aggregation_explanation = (
                    f"Escalated from {rules_attention} to {ml_level} "
                    f"based on ML signal (confidence {ml_conf:.2f} ≥ {confidence_threshold})."
                )
            else:
                aggregation_explanation = (
                    "No escalation applied. Rules remain primary. "
                    "ML signal did not exceed rules level with sufficient confidence."
                )

        if final_attention == "High":
            st.warning(f"Final attention: {final_attention}")
        elif final_attention == "Medium":
            st.info(f"Final attention: {final_attention}")
        else:
            st.success(f"Final attention: {final_attention}")

        st.caption(aggregation_explanation)

    # Export assessment record (JSON) with aggregation
    st.divider()
    st.subheader("Export assessment record (JSON)")

    assessment_record = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tool_version": TOOL_VERSION,
        },
        "inputs": st.session_state["submitted_payload"],
        "rules_output": asdict(result),
        "ml_output": ml_export,
        "aggregation": {
            "rules_attention": rules_attention,
            "final_attention": final_attention,
            "aggregation_explanation": aggregation_explanation,
            "confidence_threshold": confidence_threshold,
            "policy": {
                "rules_primary": True,
                "ml_escalation_only": True,
                "ml_applied_only_when_readiness_ready": True,
            },
        },
    }

    json_str = json.dumps(assessment_record, indent=2)

    st.download_button(
        label="Download assessment record (JSON)",
        data=json_str,
        file_name="assessment_record.json",
        mime="application/json",
    )
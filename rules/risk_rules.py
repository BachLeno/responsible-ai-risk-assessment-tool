from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# Data structures (outputs)
@dataclass
class DomainAssessment:
    level: str  # "Low" | "Medium" | "High"
    reasons: List[str]  # traceable plain-language drivers


@dataclass
class ActionItem:
    priority: str   # "High" | "Medium" | "Low"
    action: str
    rationale: str


@dataclass
class RuleResult:
    overall_attention: str

    assessment_readiness: str  # "Ready" | "Provisional" | "Insufficient information"
    unknown_fields_count: int
    critical_unknowns: List[str]

    attention_level_if_complete: str  # "Low" | "Medium" | "High"

    domain_profile: Dict[str, DomainAssessment]
    risk_categories: List[str]
    compliance_flags: List[str]
    explanations: List[str]
    recommended_actions: List[ActionItem]


# Helpers
def _add_unique(items: List[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _level_from_score(score: int) -> str:
    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


def _max_level(levels: List[str]) -> str:
    order = {"Low": 0, "Medium": 1, "High": 2}
    return max(levels, key=lambda x: order.get(x, 0))


def _is_unknown_str(value: Any) -> bool:
    """True when the UI explicitly returned Unknown-like strings."""
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip().lower()
        return s == "" or s == "unknown"
    return False


def _is_missing_multiselect(value: Any) -> bool:
    """
    Multiselects do not have an explicit 'Unknown' state in the UI, but
    an empty selection usually means the user didn't provide the information.
    """
    return isinstance(value, list) and len(value) == 0


def _collect_unknowns(payload: Dict[str, Any]) -> Tuple[int, List[str]]:
    """
    UI-aligned unknown counting:
    - Count fields that explicitly have "Unknown" as a UI option and are set to Unknown
    - Also count multiselects where empty selection implies missing information
    """
    decision = payload.get("decision_impact", {})
    data = payload.get("data_privacy", {})
    system = payload.get("system_model", {})
    gov = payload.get("governance", {})

    unknown_fields: List[str] = []

    # YES/NO/UNKNOWN fields (explicit Unknown)
    if _is_unknown_str(decision.get("appeal_possible", "Unknown")):
        unknown_fields.append("Right to contest / appeal planned")
    if _is_unknown_str(data.get("personal_data", "Unknown")):
        unknown_fields.append("Personal data")
    if _is_unknown_str(data.get("retention_defined", "Unknown")):
        unknown_fields.append("Data retention defined")
    if _is_unknown_str(data.get("cross_border_sharing", "Unknown")):
        unknown_fields.append("Cross-border sharing")
    if _is_unknown_str(system.get("monitoring_planned", "Unknown")):
        unknown_fields.append("Monitoring planned")
    if _is_unknown_str(gov.get("protected_groups_affected", "Unknown")):
        unknown_fields.append("Protected groups affected")
    if _is_unknown_str(gov.get("accountable_owner_defined", "Unknown")):
        unknown_fields.append("Accountable owner defined")

    # LOW/MED/HIGH/UNKNOWN fields (explicit Unknown)
    if _is_unknown_str(system.get("misuse_concern", "Unknown")):
        unknown_fields.append("Security / misuse concern")
    if _is_unknown_str(gov.get("bias_risk", "Unknown")):
        unknown_fields.append("Bias risk")

    # Selectboxes that include 'Unknown'
    if _is_unknown_str(system.get("model_transparency", "Unknown")):
        unknown_fields.append("Model transparency")
    if _is_unknown_str(system.get("training_data_confidence", "Unknown")):
        unknown_fields.append("Training data quality confidence")

    # Multiselects: treat empty selection as missing info
    ai_types = system.get("ai_approach_types", []) or []
    if _is_missing_multiselect(ai_types) or ("Unknown/not decided" in ai_types):
        unknown_fields.append("AI approach types")

    data_sources = data.get("data_sources", []) or []
    if _is_missing_multiselect(data_sources):
        unknown_fields.append("Data sources")

    return len(unknown_fields), unknown_fields


def _critical_unknowns(payload: Dict[str, Any]) -> List[str]:
    decision = payload.get("decision_impact", {})
    data = payload.get("data_privacy", {})
    system = payload.get("system_model", {})

    missing: List[str] = []

    # Critical early-phase fields (robust to future UI changes)
    if _is_unknown_str(decision.get("decision_type", "Unknown")):
        missing.append("Decision type")
    if _is_unknown_str(decision.get("decision_criticality", "Unknown")):
        missing.append("Decision criticality")
    if _is_unknown_str(data.get("personal_data", "Unknown")):
        missing.append("Personal data")

    ai_types = system.get("ai_approach_types", []) or []
    if _is_missing_multiselect(ai_types) or ("Unknown/not decided" in ai_types):
        missing.append("AI approach types")

    return missing


def _determine_readiness(unknown_count: int, critical_missing: List[str]) -> str:
    if critical_missing:
        return "Insufficient information"
    if unknown_count >= 4:
        return "Provisional"
    return "Ready"


def _priority_from_level(level: str) -> str:
    if level == "High":
        return "High"
    if level == "Medium":
        return "Medium"
    return "Low"


def _add_action(actions: List[ActionItem], priority: str, action: str, rationale: str) -> None:
    actions.append(ActionItem(priority=priority, action=action, rationale=rationale))


def _sort_actions(actions: List[ActionItem]) -> List[ActionItem]:
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(actions, key=lambda a: order.get(a.priority, 99))


# Main rules engine
def assess_with_rules(payload: Dict[str, Any]) -> RuleResult:
    risk_categories: List[str] = []
    compliance_flags: List[str] = []
    explanations: List[str] = []

    basics = payload.get("use_case_basics", {})
    decision = payload.get("decision_impact", {})
    data = payload.get("data_privacy", {})
    system = payload.get("system_model", {})
    gov = payload.get("governance", {})

    domain = basics.get("domain_context", "Unknown")
    decision_type = decision.get("decision_type", "Unknown")
    criticality = decision.get("decision_criticality", "Unknown")
    human_loop = decision.get("human_in_loop", "Unknown")

    personal_data = data.get("personal_data", "Unknown")
    sensitive_types = data.get("sensitive_data_types", []) or []
    cross_border = data.get("cross_border_sharing", "Unknown")

    ai_types = system.get("ai_approach_types", []) or []
    transparency = system.get("model_transparency", "Unknown")
    misuse = system.get("misuse_concern", "Unknown")

    bias_risk = gov.get("bias_risk", "Unknown")
    protected_groups = gov.get("protected_groups_affected", "Unknown")

    # Uncertainty / information quality
    unknown_fields_count, _unknown_fields = _collect_unknowns(payload)
    critical_missing = _critical_unknowns(payload)
    assessment_readiness = _determine_readiness(unknown_fields_count, critical_missing)

    if assessment_readiness != "Ready":
        explanations.append(
            "Information quality: some key fields are unknown. "
            "The headline is marked provisional/insufficient to avoid false certainty."
        )
        if critical_missing:
            explanations.append("Critical unknowns: " + ", ".join(critical_missing))

    # Rule flags
    if personal_data == "Yes":
        _add_unique(risk_categories, "Privacy & Data Protection")
        _add_unique(compliance_flags, "GDPR signal: personal data processing indicated")
        explanations.append("Personal data marked 'Yes' → consider GDPR-related controls and documentation.")

    if len(sensitive_types) > 0:
        _add_unique(risk_categories, "Privacy & Data Protection")
        _add_unique(compliance_flags, "GDPR signal: special category (sensitive) data indicated")
        explanations.append("Sensitive data selected → increased privacy governance expectations (indicative).")

    if cross_border == "Yes":
        _add_unique(compliance_flags, "Data transfer/sharing signal: cross-border or external sharing expected")
        explanations.append("Cross-border/external sharing marked 'Yes' → consider transfer/sharing safeguards (indicative).")

    high_impact_domains = {
        "HR / recruitment",
        "Finance / credit / insurance",
        "Healthcare",
        "Education",
        "Security / law enforcement",
        "Public sector",
    }

    if domain in high_impact_domains and "High" in str(criticality):
        _add_unique(compliance_flags, "EU AI Act signal: high-impact context suspected (indicative)")
        explanations.append("High-impact domain + high criticality → elevated governance expectations likely (indicative).")

    if bias_risk in {"Medium", "High"} or protected_groups == "Yes":
        _add_unique(risk_categories, "Fairness & Non-discrimination")
        explanations.append("Bias/protected-group indicator → fairness/discrimination category flagged.")

    if "Low" in str(transparency) or "black-box" in str(transparency).lower():
        _add_unique(risk_categories, "Transparency & Explainability")
        explanations.append("Low/black-box transparency → higher explainability needs.")

    if misuse in {"Medium", "High"}:
        _add_unique(risk_categories, "Security & Misuse")
        explanations.append("Misuse concern medium/high → consider misuse scenarios and safeguards.")

    if "Automated decision" in str(decision_type) or "Automated action" in str(decision_type):
        _add_unique(risk_categories, "Human Oversight & Accountability")
        explanations.append("High automation selected → oversight/accountability measures become more important.")

    if human_loop == "No":
        _add_unique(risk_categories, "Human Oversight & Accountability")
        explanations.append("Human-in-the-loop is 'No' → increases need for accountability/oversight controls.")

    if "Generative AI / LLM" in ai_types:
        _add_unique(risk_categories, "Transparency & Explainability")
        _add_unique(risk_categories, "Security & Misuse")
        explanations.append("Generative AI selected → typical concerns include hallucinations, transparency, and misuse.")

    # Domain risk profile
    domain_profile: Dict[str, DomainAssessment] = {}

    privacy_score = 0
    privacy_reasons: List[str] = []
    if personal_data == "Yes":
        privacy_score += 2
        privacy_reasons.append("Personal data processed.")
    if len(sensitive_types) > 0:
        privacy_score += 2
        privacy_reasons.append("Sensitive (special category) data indicated.")
    if cross_border == "Yes":
        privacy_score += 1
        privacy_reasons.append("Cross-border/external sharing expected.")

    domain_profile["Privacy & Data Protection"] = DomainAssessment(
        level=_level_from_score(privacy_score),
        reasons=privacy_reasons or ["No privacy indicators triggered from current inputs."],
    )

    fairness_score = 0
    fairness_reasons: List[str] = []
    if bias_risk in {"Medium", "High"}:
        fairness_score += 2 if bias_risk == "High" else 1
        fairness_reasons.append(f"Bias risk marked {bias_risk}.")
    if protected_groups == "Yes":
        fairness_score += 2
        fairness_reasons.append("Protected groups potentially affected.")

    domain_profile["Fairness & Non-discrimination"] = DomainAssessment(
        level=_level_from_score(fairness_score),
        reasons=fairness_reasons or ["No fairness indicators triggered from current inputs."],
    )

    transparency_score = 0
    transparency_reasons: List[str] = []
    if "Low" in str(transparency) or "black-box" in str(transparency).lower():
        transparency_score += 2
        transparency_reasons.append("Low / black-box transparency selected.")
    if "Generative AI / LLM" in ai_types:
        transparency_score += 1
        transparency_reasons.append("Generative AI selected (typical explainability challenges).")

    domain_profile["Transparency & Explainability"] = DomainAssessment(
        level=_level_from_score(transparency_score),
        reasons=transparency_reasons or ["No transparency indicators triggered from current inputs."],
    )

    oversight_score = 0
    oversight_reasons: List[str] = []
    if "Automated decision" in str(decision_type) or "Automated action" in str(decision_type):
        oversight_score += 2
        oversight_reasons.append("High automation selected.")
    if human_loop == "No":
        oversight_score += 2
        oversight_reasons.append("No human-in-the-loop.")
    elif "Sometimes" in str(human_loop):
        oversight_score += 1
        oversight_reasons.append("Human oversight is exception-based.")

    domain_profile["Human Oversight & Accountability"] = DomainAssessment(
        level=_level_from_score(oversight_score),
        reasons=oversight_reasons or ["No oversight indicators triggered from current inputs."],
    )

    security_score = 0
    security_reasons: List[str] = []
    if misuse in {"Medium", "High"}:
        security_score += 2 if misuse == "High" else 1
        security_reasons.append(f"Misuse concern marked {misuse}.")
    if "Generative AI / LLM" in ai_types:
        security_score += 1
        security_reasons.append("Generative AI selected (typical misuse risks).")

    domain_profile["Security & Misuse"] = DomainAssessment(
        level=_level_from_score(security_score),
        reasons=security_reasons or ["No security/misuse indicators triggered from current inputs."],
    )

    attention_level_if_complete = _max_level([d.level for d in domain_profile.values()])

    if assessment_readiness == "Insufficient information":
        overall_attention = "Insufficient information"
    elif assessment_readiness == "Provisional":
        overall_attention = "Provisional"
    else:
        overall_attention = attention_level_if_complete

    # Actionable governance checklist
    actions: List[ActionItem] = []

    if assessment_readiness == "Insufficient information":
        _add_action(
            actions,
            "High",
            "Complete critical missing information",
            "The assessment cannot provide a reliable headline until critical fields are answered: "
            + ", ".join(critical_missing),
        )

    if assessment_readiness == "Provisional":
        _add_action(
            actions,
            "High",
            "Reduce unknown fields and confirm key assumptions",
            "Many key inputs are still unknown. Completing the form improves assessment reliability.",
        )

    privacy_level = domain_profile["Privacy & Data Protection"].level
    if privacy_level in {"Medium", "High"}:
        _add_action(
            actions,
            _priority_from_level(privacy_level),
            "Clarify data processing purpose and lawful basis (GDPR)",
            "Personal and/or sensitive data signals increase privacy governance needs.",
        )
        _add_action(
            actions,
            _priority_from_level(privacy_level),
            "Define data retention, access control, and deletion plan",
            "Early documentation reduces later compliance risk and supports accountability.",
        )
        if privacy_level == "High":
            _add_action(
                actions,
                "High",
                "Consider a Data Protection Impact Assessment (DPIA) and consult the Data Protection Officer (DPO)",
                "High privacy risk indicators suggest DPIA-style assessment may be required (context-dependent).",
            )
    if cross_border == "Yes":
        _add_action(
            actions,
            "Medium",
            "Review data sharing / transfer mechanisms and processor agreements",
            "Cross-border or external sharing requires governance and contractual safeguards (indicative).",
        )

    fairness_level = domain_profile["Fairness & Non-discrimination"].level
    if fairness_level in {"Medium", "High"}:
        _add_action(
            actions,
            _priority_from_level(fairness_level),
            "Define fairness criteria and plan bias testing",
            "Bias/protected group signals indicate need for fairness evaluation and documentation.",
        )
        _add_action(
            actions,
            _priority_from_level(fairness_level),
            "Document affected stakeholders and potential disparate impact",
            "Early stakeholder mapping supports responsible governance and mitigation planning.",
        )

    transparency_level = domain_profile["Transparency & Explainability"].level
    if transparency_level in {"Medium", "High"}:
        _add_action(
            actions,
            _priority_from_level(transparency_level),
            "Prepare transparency documentation (model purpose, data sources, limitations)",
            "Low transparency / complex models increase the need for explainability and clear communication.",
        )
        _add_action(
            actions,
            _priority_from_level(transparency_level),
            "Define how decisions will be explained to users/stakeholders",
            "Explainability planning supports accountability and user trust (context-dependent).",
        )

    oversight_level = domain_profile["Human Oversight & Accountability"].level
    if oversight_level in {"Medium", "High"}:
        _add_action(
            actions,
            _priority_from_level(oversight_level),
            "Define human oversight role and escalation path",
            "Automation signals increase the need for clear accountability and human review procedures.",
        )
        _add_action(
            actions,
            _priority_from_level(oversight_level),
            "Define an appeal / review mechanism (where appropriate)",
            "For impactful decisions, governance commonly includes a path for review and contestability.",
        )

    security_level = domain_profile["Security & Misuse"].level
    if security_level in {"Medium", "High"}:
        _add_action(
            actions,
            _priority_from_level(security_level),
            "Perform misuse/threat scenario review",
            "Misuse concern signals indicate the need to identify abuse cases and safeguards.",
        )
        _add_action(
            actions,
            _priority_from_level(security_level),
            "Set monitoring and incident response plan",
            "Governance benefits from clear monitoring responsibilities and response procedures.",
        )

    if domain in high_impact_domains and "High" in str(criticality):
        _add_action(
            actions,
            "High",
            "Escalate to compliance/governance review (high-impact context)",
            "High-impact domain + high criticality suggests stronger governance review is appropriate.",
        )

    actions = _sort_actions(actions)

    if not risk_categories:
        risk_categories = ["No major risk category flagged (based on current inputs)"]
    if not compliance_flags:
        compliance_flags = ["No major compliance signals flagged (based on current inputs)"]
    if not explanations:
        explanations = ["No specific rule explanations were triggered."]

    return RuleResult(
        overall_attention=overall_attention,
        assessment_readiness=assessment_readiness,
        unknown_fields_count=unknown_fields_count,
        critical_unknowns=critical_missing,
        attention_level_if_complete=attention_level_if_complete,
        domain_profile=domain_profile,
        risk_categories=risk_categories,
        compliance_flags=compliance_flags,
        explanations=explanations,
        recommended_actions=actions,
    )
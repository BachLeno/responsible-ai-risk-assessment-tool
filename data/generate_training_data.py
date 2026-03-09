from pathlib import Path
import random
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "training_data.csv"

RANDOM_SEED = 42


LOW_BASE = [
    "Internal dashboard summarizing team productivity metrics for managers.",
    "Tool that recommends internal training courses based on previous learning history.",
    "Automation that routes internal support tickets to the correct team.",
    "Internal analytics report that highlights trends in product sales.",
    "System that drafts email replies for common customer support questions (human edits before sending).",
]

MEDIUM_BASE = [
    "AI system supporting HR to shortlist CVs before interviews.",
    "Fraud detection model that flags suspicious transactions for analyst review.",
    "Credit scoring tool used by financial staff to support lending decisions.",
    "Healthcare triage assistant suggesting priority levels for nurse review.",
    "Supplier risk scoring model ranking vendors for procurement decision-making.",
]

HIGH_BASE = [
    "Automated credit approval system making final decisions without routine human review.",
    "AI system determining insurance eligibility automatically for most cases.",
    "Automated hiring rejection system where applicants have no clear appeal process.",
    "Medical diagnosis system recommending treatments as final decisions in practice.",
    "Public security system using biometric identification to identify individuals.",
]

PRIVACY = [
    "No personal data is processed.",
    "Limited personal data is processed, such as user IDs and contact details.",
    "The system processes personal data including names, contact details, and account history.",
    "Sensitive data such as health or biometric data may be included.",
]

FAIRNESS = [
    "Fairness risks are considered low based on current understanding.",
    "Bias testing has not yet been performed.",
    "There is potential impact on protected groups depending on deployment context.",
]

SECURITY = [
    "Security controls are in place.",
    "Security controls are partially implemented and monitoring is limited.",
    "The system may be vulnerable to misuse or adversarial manipulation.",
]

HUMAN = [
    "Final decisions are reviewed by a human.",
    "Human oversight is limited to spot checks or exceptions.",
    "No human review is performed before outcomes are applied.",
]

# Overlap terms that appear across classes (to reduce easy keyword separation)
OVERLAP_SNIPPETS = [
    "The output is used as a risk assessment signal.",
    "The system generates a score used for prioritization.",
    "Monitoring and periodic reviews are planned after deployment.",
    "The system supports eligibility decisions in some workflows.",
    "The tool is partially automated and integrates into existing processes.",
]

# Paraphrases for strong giveaway phrases
PARAPHRASES = [
    ("Fully automated", "Automated end-to-end"),
    ("no human review", "without routine human checks"),
    ("determining insurance eligibility", "supporting insurance eligibility decisions"),
    ("Automated hiring rejection", "Hiring outcome automation"),
]


def _apply_paraphrases(text: str) -> str:
    for a, b in PARAPHRASES:
        if a in text and random.random() < 0.7:
            text = text.replace(a, b)
    return text


def build_dataset(samples_per_class: int = 60) -> pd.DataFrame:
    rows = []

    for label, base_pool in [
        ("Low", LOW_BASE),
        ("Medium", MEDIUM_BASE),
        ("High", HIGH_BASE),
    ]:
        for _ in range(samples_per_class):
            base = random.choice(base_pool)
            privacy = random.choice(PRIVACY)
            fairness = random.choice(FAIRNESS)
            security = random.choice(SECURITY)
            human = random.choice(HUMAN)

            # Add overlap snippet with high probability so all classes share vocabulary
            overlap = random.choice(OVERLAP_SNIPPETS) if random.random() < 0.8 else ""

            parts = [base, privacy, fairness, security, human, overlap]
            parts = [p for p in parts if p]  # drop empty

            # Randomly drop one signal sentence (structure variation)
            if len(parts) >= 5 and random.random() < 0.10:
                drop_idx = random.randint(1, 4)  # drop one of privacy/fairness/security/human
                parts.pop(drop_idx)

            text = " ".join(parts)
            text = _apply_paraphrases(text)

            # Controlled "edge cases"
            if label == "Low":
                # Some Low cases mention basic personal data + "eligibility", but low impact and internal
                if random.random() < 0.20:
                    text += " This is used for internal eligibility checks for training access only."
                if random.random() < 0.20:
                    text += " The system stores basic user account information and audit logs."

                # Keep Low clearly low-impact by sometimes reinforcing limited consequences
                if random.random() < 0.30:
                    text += " The outcome does not directly affect rights, employment, or finances."

            elif label == "Medium":
                # Some Medium cases resemble High: more automation language + higher stakes phrasing
                if random.random() < 0.30:
                    text += " Decisions may influence access to services, but a reviewer can override outcomes."
                if random.random() < 0.25:
                    text += " Appeals are possible but the process is not fully defined yet."
                if random.random() < 0.25:
                    text += " The system is used at moderate scale across departments."

            elif label == "High":
                # Soften some High wording to mimic real org descriptions
                if random.random() < 0.25:
                    text += " Human intervention is possible but uncommon in day-to-day operations."
                if random.random() < 0.20:
                    text += " Decisions can significantly affect individuals in employment, finance, health, or rights."
                if random.random() < 0.20:
                    text += " The system operates at large scale and outcomes may be difficult to contest."

            rows.append({"text": text, "label": label})

    random.shuffle(rows)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    df = build_dataset(samples_per_class=60)  # 180 total
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset generated at: {OUTPUT_PATH}")
    print(df["label"].value_counts())
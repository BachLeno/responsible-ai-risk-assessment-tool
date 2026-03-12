# Responsible AI Risk Assessment Tool

Prototype decision-support tool for early-phase Responsible AI risk and compliance assessment, developed as part of a Bachelor's thesis at Häme University of Applied Sciences (HAMK).

## Overview

This project is a proof-of-concept web application for structuring early-phase AI governance assessments. Users describe an AI use case through a guided Streamlit form, after which the tool:

- evaluates governance signals with explicit rule-based logic
- analyzes the free-text description with a lightweight ML classifier
- combines both signals into a final hybrid attention result
- produces explanations, domain-level risk signals, compliance flags, recommended actions, and a downloadable JSON record

The prototype is designed to support reflection, consistency, and traceability in early AI project planning. It is **not** a legal compliance automation system and should not be used as legal advice or as a substitute for formal review.

## What the application assesses

The interface collects structured input across five sections:

1. **Use case basics**
2. **Decision & impact characteristics**
3. **Data & privacy signals**
4. **System & governance signals**
5. **Responsible AI & documentation signals**

From these inputs, the rules engine builds a domain risk profile covering:

- Privacy & Data Protection
- Fairness & Non-discrimination
- Transparency & Explainability
- Human Oversight & Accountability
- Security & Misuse

## Assessment outputs

The application produces:

- an overall rules-based attention level
- an assessment readiness status:
  - `Ready`
  - `Provisional`
  - `Insufficient information`
- a domain risk profile with plain-language reasons
- flagged risk categories
- indicative compliance signals
- recommended governance actions
- an optional ML supporting signal from the use-case description
- a final hybrid attention decision
- an exportable JSON assessment record

### Hybrid decision logic

The rules engine is the primary decision mechanism.

The ML classifier may **escalate** the final attention level only when:

- the rules-based readiness status is `Ready`
- the ML-predicted level is higher than the rules-based level
- the ML confidence is at least `0.75`

Otherwise, the rules-based result remains unchanged.

## Repository structure

```text
responsible-ai-risk-assessment-tool/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── data/
│   ├── generate_training_data.py
│   └── training_data.csv
├── ml/
│   ├── text_classifier.py
│   ├── model.joblib
│   └── metrics.json
└── rules/
    └── risk_rules.py
```

## Tech stack

- Python
- Streamlit
- pandas
- numpy
- scikit-learn
- joblib

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/BachLeno/responsible-ai-risk-assessment-tool.git
cd responsible-ai-risk-assessment-tool
```

### 2. Create and activate a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

By default, Streamlit serves the app locally at:

```text
http://localhost:8501
```

## Machine learning workflow

The repository includes a pre-trained model in `ml/model.joblib`, so the app can run without retraining.

### Generate synthetic training data

```bash
python data/generate_training_data.py
```

### Train and evaluate the classifier

```bash
python -m ml.text_classifier train-eval
```

### Other available CLI commands

```bash
python -m ml.text_classifier train
python -m ml.text_classifier eval
```

The training/evaluation workflow updates:

- `ml/model.joblib`
- `ml/metrics.json`

## Methodology notes

- The ML component uses TF-IDF features with logistic regression.
- The classifier predicts one of three labels: `Low`, `Medium`, or `High`.
- The training data is synthetic and intended for prototype demonstration only.
- The rules engine remains the primary governance assessment mechanism.

## Limitations

- This is a prototype, not a production governance platform.
- The rule logic is intentionally simplified.
- The ML model is trained on synthetic examples rather than real organizational cases.
- Results are indicative and should be reviewed by human stakeholders.
- The thesis evaluation used a limited scenario-based comparison rather than real-world deployment testing.

## Thesis context

This repository supports the Bachelor's thesis **Support Tool for Assessing Responsible AI Risk and Compliance**.

The thesis describes a prototype that combines rule-based assessment logic with a lightweight machine learning component, implemented as a Streamlit web application and evaluated using three constructed scenarios. The goal is to support structured early-phase AI risk and compliance assessment rather than automate formal compliance decisions.

## Author

Leno Bach  
Bachelor's Thesis  
Degree Programme in Computer Applications  
Häme University of Applied Sciences (HAMK)

## License

This project is licensed under the MIT License. See `LICENSE` for details.
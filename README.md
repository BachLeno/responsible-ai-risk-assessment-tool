# Responsible AI Risk Assessment Tool

## Overview

This repository contains a prototype **Responsible AI Risk Assessment Tool** developed as part of a **Bachelor's thesis** in the Degree Programme in Computer Applications at Häme University of Applied Sciences (HAMK).

The tool supports the **early-stage assessment of AI systems** by identifying potential governance risks and compliance considerations related to responsible AI development. It combines:

- a **rule-based governance assessment module**
- a **lightweight machine learning classifier**
- a **web-based user interface implemented with Streamlit**

The objective of the prototype is to demonstrate how structured input parameters and automated analysis can support the **initial evaluation of AI use cases from a Responsible AI perspective**.

> **Note:**  
> This tool is a research prototype developed for academic purposes. It should not be used as a substitute for formal regulatory or legal assessments.

## Features

The prototype provides the following functionality:

- Structured input form for describing an AI use case
- Rule-based evaluation of governance signals
- Domain-level Responsible AI risk profile
- Identification of potential compliance signals
- Machine learning–assisted risk classification
- Explanations of detected risks
- Suggested governance actions

The tool focuses on common Responsible AI dimensions, including:

- Privacy and data protection  
- Fairness and bias  
- Transparency and explainability  
- Accountability and governance  
- Security and misuse risks  

## Repository Structure

```text
responsible-ai-risk-assessment-tool/
│
├── app.py                # Main Streamlit application & GUI
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── LICENSE               # License information
├── .gitignore            # Files excluded from version control
│
├── data/
│   ├── generate_training_data.py  # Script for synthetic data generation
│   └── training_data.csv          # Dataset for model training
│
├── ml/
│   ├── text_classifier.py         # ML classification logic
│   ├── model.joblib               # Pre-trained model file
│   └── metrics.json               # Model evaluation metrics
│
└── rules/
    └── risk_rules.py              # Rule-based governance logic
```

### Main Components

**app.py**  
Main Streamlit application providing the graphical user interface and orchestrating the assessment workflow.

**rules/risk_rules.py**  
Contains the rule-based governance logic used to evaluate structured input parameters and generate risk signals.

**ml/text_classifier.py**  
Implements the machine learning classifier used to categorize AI use cases based on textual descriptions.

**data/generate_training_data.py**  
Script used to generate synthetic training data for the classifier.

**data/training_data.csv**  
Dataset used for training the machine learning model.

**ml/model.joblib**  
Pre-trained machine learning model used by the application.

**ml/metrics.json**  
Evaluation metrics of the trained model.

## Setup and Installation

The following steps describe how to install and run the Responsible AI Risk Assessment Tool locally.

## 1. Clone the Repository

```bash
git clone https://github.com/BachLeno/responsible-ai-risk-assessment-tool.git  
cd responsible-ai-risk-assessment-tool
```

## 2. Create a Virtual Environment

It is recommended to use a Python virtual environment to manage project dependencies and avoid conflicts with other Python installations.

### Windows

```bash
python -m venv venv  
venv\Scripts\activate  
```

### macOS / Linux

```bash
python3 -m venv venv  
source venv/bin/activate  
```

After activation, the terminal prompt should show the name of the virtual environment (for example `(venv)`).

## 3. Install Dependencies

Install the required Python libraries using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

This will install all necessary packages, including:

- Streamlit  
- pandas  
- numpy  
- scikit-learn  
- joblib  

## Running the Application

The tool is implemented using **Streamlit**.

Start the application by running:

```bash
streamlit run app.py
```

Streamlit will start a local development server and automatically open the application in your default web browser.

If the browser does not open automatically, the application can usually be accessed at:

http://localhost:8501

The interface allows users to enter information about an AI system and generate a structured governance risk assessment.

## Machine Learning Component

The repository includes a **pre-trained model** (`ml/model.joblib`) so the application can run immediately after installation.

The machine learning component is used to support the classification of AI use cases based on textual descriptions.

## Optional: Retraining the Machine Learning Model

If desired, the training dataset and model can be regenerated.

### Generate Synthetic Training Data

```bash
python data/generate_training_data.py
```

### Train the Classifier

```bash
python ml/text_classifier.py
```

After training, the following files will be generated:

```bash
ml/model.joblib  
ml/metrics.json  
```

These files are used by the application to perform the text-based risk classification.

## Methodological Notes

The machine learning model is trained using **synthetic training data generated specifically for this prototype**.

Therefore, the ML classifier should be interpreted as a **supporting signal rather than a production-grade risk prediction model**.

The rule-based logic remains the primary mechanism for identifying governance risks.

## Limitations

This prototype has several limitations:

- The rule-based assessment logic is simplified
- The machine learning model is trained on synthetic data
- The tool has not been validated in real organizational environments
- The results should be interpreted as **decision-support guidance rather than formal compliance evaluation**

## Technologies Used

- Python  
- Streamlit  
- pandas  
- numpy  
- scikit-learn  
- joblib

## Author

Leno Bach  
Bachelor's Thesis  
Degree Programme in Computer Applications  
Häme University of Applied Sciences (HAMK)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

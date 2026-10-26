
import numpy as np
import gradio as gr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# EDUCATIONAL PROTOTYPE ONLY.
# All training records below are synthetic, not bank customer data.

rng = np.random.default_rng(42)
n = 3000

income = rng.integers(15000, 150001, n)
credit = rng.integers(300, 901, n)
loan = rng.integers(20000, 1500001, n)
emi = rng.integers(0, 50001, n)
experience = rng.integers(0, 31, n)

X = np.column_stack(
    [income, credit, loan, emi, experience]
)

# Simplified teaching labels, not real lending rules.
labels = (
    (credit >= 650)
    & (income >= 25000)
    & (loan * 0.015 + emi <= income * 0.45)
    & (experience >= 1)
).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42,
    stratify=labels
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)

test_accuracy = accuracy_score(
    y_test, model.predict(X_test)
)

def predict(income, credit, loan, emi, experience):
    values = [income, credit, loan, emi, experience]

    if any(v is None for v in values):
        return "Please enter all five values."

    if income <= 0 or loan <= 0 or emi < 0:
        return "Income and loan must be positive; EMI cannot be negative."

    if not 300 <= credit <= 900:
        return "Enter a credit score between 300 and 900."

    if experience < 0 or experience > 60:
        return "Enter employment experience between 0 and 60 years."

    features = np.array([values], dtype=float)
    prediction = int(model.predict(features)[0])
    confidence = float(
        model.predict_proba(features)[0][prediction]
    ) * 100

    result = (
        "Prototype result: Positive indication"
        if prediction == 1
        else "Prototype result: Negative indication"
    )

    return f"""
## {result}

- Monthly income: ₹{income:,.0f}
- Credit score: {credit:.0f}
- Requested loan: ₹{loan:,.0f}
- Existing monthly EMI: ₹{emi:,.0f}
- Employment experience: {experience:.0f} years
- Model confidence for this predicted class: {confidence:.1f}%

**Important:** This is an educational prediction from a model
trained on synthetic data. It is NOT a bank approval or rejection,
and it must not be used to make real lending decisions.
"""

with gr.Blocks(title="AI BFSI Loan Eligibility Checker") as demo:
    gr.Markdown("""
    # 🏦 AI-Powered BFSI Loan Eligibility Checker

    An educational machine-learning prototype demonstrating
    how applicant information can be used for a prediction.

    **Demo only:** Uses synthetic training data, not real bank data.
    """)

    with gr.Row():
        income_input = gr.Number(
            label="Monthly income (₹)", value=50000
        )
        credit_input = gr.Number(
            label="Credit score (300–900)", value=720
        )

    with gr.Row():
        loan_input = gr.Number(
            label="Requested loan amount (₹)", value=300000
        )
        emi_input = gr.Number(
            label="Existing monthly EMI (₹)", value=5000
        )

    experience_input = gr.Number(
        label="Employment experience (years)", value=3
    )

    check_button = gr.Button("Check Prototype Result")
    output = gr.Markdown()

    check_button.click(
        fn=predict,
        inputs=[
            income_input, credit_input, loan_input,
            emi_input, experience_input
        ],
        outputs=output
    )

    gr.Markdown(
        f"**Internal holdout accuracy:** {test_accuracy:.1%} "
        "(synthetic test data only; not evidence of real-world "
        "banking performance)."
    )

if __name__ == "__main__":
    demo.launch()
# 🏦 AI-Powered BFSI Loan Eligibility Checker

**Developer:** Sakshi Chavan  
**Domain:** Banking, Financial Services, and Insurance (BFSI)  
**Tech Stack:** Python 3.10+, Scikit-Learn, Gradio, Pandas, NumPy, Joblib  

---

## 📌 Project Overview
This capstone project implements an automated, end-to-end Machine Learning solution designed for real-time loan underwriting and credit eligibility assessment in the BFSI sector. Using a supervised **Random Forest Classifier**, the system analyzes multi-factor applicant financial profiles (income metrics, credit history, requested loan size) to output an instant decision alongside risk confidence scores.

---

## 🚀 Key Features
- **Instant Risk Assessment:** Evaluates default likelihood in real time using tabular applicant data.
- **Explainable Decision Confidence:** Displays explicit approval percentage and default probability scores.
- **Zero-Footprint Mobile UI:** Powered by Gradio for cross-device compatibility (iOS, Android, Desktop).
- **Automated Feature Engineering:** Combines applicant and co-applicant incomes into total debt capacity parameters.

---

## 🛠️ Project Structure
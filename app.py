
import numpy as np
import gradio as gr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Educational prototype using synthetic data only
rng = np.random.default_rng(42)
n = 3000

income = rng.integers(15000, 150001, n)
credit = rng.integers(300, 901, n)
loan = rng.integers(20000, 1500001, n)
emi = rng.integers(0, 50001, n)
experience = rng.integers(0, 31, n)

X = np.column_stack([income, credit, loan, emi, experience])

# Simplified teaching labels, not real bank lending rules
y = (
    (credit >= 650)
    & (income >= 25000)
    & (loan * 0.015 + emi <= income * 0.45)
    & (experience >= 1)
).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))


def predict(income, credit, loan, emi, experience):
    values = [income, credit, loan, emi, experience]

    if any(v is None for v in values):
        return "Please enter all five values."

    if income <= 0 or loan <= 0 or emi < 0:
        return "Income and loan must be positive; EMI cannot be negative."

    if not 300 <= credit <= 900:
        return "Credit score must be between 300 and 900."

    if not 0 <= experience <= 60:
        return "Experience must be between 0 and 60 years."

    features = np.array([values], dtype=float)
    prediction = int(model.predict(features)[0])
    confidence = model.predict_proba(features)[0][prediction] * 100

    result = (
        "Positive indication"
        if prediction == 1
        else "Negative indication"
    )

    return f"""
## {result}

- Monthly income: ₹{income:,.0f}
- Credit score: {credit}
- Requested loan: ₹{loan:,.0f}
- Existing monthly EMI: ₹{emi:,.0f}
- Employment experience: {experience} years
- Model confidence for this class: {confidence:.1f}%

**Disclaimer:** This educational model uses synthetic data.
This is not a real bank loan approval or rejection.
"""


with gr.Blocks(title="AI BFSI Loan Eligibility Checker") as demo:
    gr.Markdown("""
    # 🏦 AI-Powered BFSI Loan Eligibility Checker

    Enter applicant details to see an educational ML prediction.

    **Demo only:** This model uses generated sample data,
    not real bank customer records.
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

    button = gr.Button("Check Eligibility Indication")
    output = gr.Markdown()

    button.click(
        predict,
        inputs=[
            income_input,
            credit_input,
            loan_input,
            emi_input,
            experience_input
        ],
        outputs=output
    )

    gr.Markdown(
        f"**Holdout accuracy on synthetic data:** {accuracy:.1%}. "
        "This does not establish real-world banking performance."
    )

if __name__ == "__main__":
    demo.launch()
  

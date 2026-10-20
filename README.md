import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import gradio as gr

# Train model on mock BFSI loan data
np.random.seed(42)
n_samples = 500

df = pd.DataFrame({
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Married': np.random.choice(['Yes', 'No'], n_samples),
    'Credit_History': np.random.choice([1.0, 0.0], n_samples, p=[0.8, 0.2]),
    'LoanAmount': np.random.randint(50, 500, n_samples),
    'ApplicantIncome': np.random.randint(2000, 20000, n_samples),
    'CoapplicantIncome': np.random.randint(0, 10000, n_samples),
})
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Loan_Status'] = np.where((df['Credit_History'] == 1.0) & (df['TotalIncome'] > (df['LoanAmount'] * 25)), 1, 0)

X = pd.get_dummies(df[['Gender', 'Married', 'Credit_History', 'LoanAmount', 'TotalIncome']], drop_first=True)
y = df['Loan_Status']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
model_columns = X.columns

def predict_loan(gender, married, credit_history, loan_amount, applicant_income, coapplicant_income):
    total_income = applicant_income + coapplicant_income
    input_df = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Credit_History': [float(credit_history)],
        'LoanAmount': [loan_amount],
        'TotalIncome': [total_income]
    })
    
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    if pred == 1:
        return f"🎉 Loan APPROVED! (Confidence: {prob*100:.1f}%)"
    else:
        return f"❌ Loan REJECTED (Risk Score: {(1-prob)*100:.1f}%)"

demo = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender", value="Male"),
        gr.Radio(["Yes", "No"], label="Married", value="Yes"),
        gr.Radio([1.0, 0.0], label="Credit History (1.0 = Good, 0.0 = Bad)", value=1.0),
        gr.Slider(10, 1000, value=120, label="Loan Amount ($1000s)"),
        gr.Number(value=5000, label="Applicant Monthly Income ($)"),
        gr.Number(value=0, label="Co-applicant Monthly Income ($)")
    ],
    outputs=gr.Textbox(label="AI Decision Result"),
    title="🏦 AI-Powered BFSI Loan Eligibility Checker",
    description="Developed by Sakshi Chavan"
)

if __name__ == "__main__":
    demo.launch()
import pandas as pd
import streamlit as st
import pickle
import shap
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv
import os
import pandas as pd

# Use the port specified by Render, or default to 10000
port = os.getenv("PORT", 10000)

# Load environment variables
load_dotenv()

# Load trained model
with open('xgb_model-SMOTE.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataset
data = pd.read_csv('Financial Market Data.csv', parse_dates=['Data'])

# Manually define stock and bond options
stock_options = ['MXUS', 'MXEU', 'MXJP', 'MXBR', 'MXRU', 'MXIN','MXCN','','','', ]  # stock column names
bond_options = ['LF94TRUU', 'GTITL30YR', 'USGG30YR', 'GT10', 'USGG2YR', 'US0001M','GTDEM30Y','GTDEM10Y','GTDEM2Y','GTITL30YR','GTITL10Y','GTITL2Y','GTJPY30YR','GTJPY10Y','GTJPY2Y','GTGBP30Y','GTGBP20Y','GTGBP2Y','LUMSTRUU', 'LMBITR','LUACTRUU','LF98TRUU','LG30TRUU','LP01TREU','EMUSTRUU',]  # bond column names

# Default stock and bond selections
default_stocks = ['MXUS']
default_bonds = ['LF94TRUU']

# Sidebar for stock and bond selection
selected_stocks = st.sidebar.multiselect("Select Stocks", stock_options, default=default_stocks)
selected_bonds = st.sidebar.multiselect("Select Bonds", bond_options, default=default_bonds)

# Define year-to-row mapping for the dataset
year_to_row = {
    2000: 50, 2001: 102, 2002: 155, 2003: 207, 2004: 259, 2005: 311,
    2006: 363, 2007: 415, 2008: 468, 2009: 520, 2010: 572, 2011: 624,
    2012: 676, 2013: 727, 2014: 781, 2015: 834, 2016: 885, 2017: 937,
    2018: 989, 2019: 1042, 2020: 1094, 2021: 1110,
}

# Sidebar for other inputs
st.sidebar.header("Portfolio Simulation Options")
selected_year = st.sidebar.selectbox("Select Year", list(year_to_row.keys()))
starting_value = st.sidebar.number_input(
    "Starting Portfolio Value ($)", min_value=1000, value=10000
)

# Default allocations
anomaly_allocation = {"cash": 70, "bonds": 20, "stocks": 10}
normal_allocation = {"cash": 10, "bonds": 20, "stocks": 70}

# Custom allocations
custom_anomaly_allocation = {
    "cash": st.sidebar.slider("Anomaly Cash Allocation (%)", 0, 100, anomaly_allocation["cash"]),
    "bonds": st.sidebar.slider("Anomaly Bond Allocation (%)", 0, 100, anomaly_allocation["bonds"]),
    "stocks": st.sidebar.slider("Anomaly Stock Allocation (%)", 0, 100, anomaly_allocation["stocks"]),
}

custom_normal_allocation = {
    "cash": st.sidebar.slider("Normal Cash Allocation (%)", 0, 100, normal_allocation["cash"]),
    "bonds": st.sidebar.slider("Normal Bond Allocation (%)", 0, 100, normal_allocation["bonds"]),
    "stocks": st.sidebar.slider("Normal Stock Allocation (%)", 0, 100, normal_allocation["stocks"]),
}

# Filter dataset by selected year
filtered_data = data.iloc[:year_to_row[selected_year]]

# Generate predictions
X_new = filtered_data.iloc[:, 2:]  # Adjust as necessary for correct feature columns
predictions = model.predict(X_new)
crisis_threshold = 0.7
filtered_data['predicted_crisis'] = (predictions > crisis_threshold).astype(int)

# Portfolio simulation
def simulate_strategy(data, stocks, bonds, starting_value):
    portfolio_value = starting_value
    portfolio_history = []
    prev_row = None

    for _, row in data.iterrows():
        # Determine allocation (custom or default)
        allocation = (
            custom_anomaly_allocation if row['predicted_crisis'] == 1 else custom_normal_allocation
        ) if any(value != default for value, default in zip(
            custom_anomaly_allocation.values(), anomaly_allocation.values()
        )) or any(value != default for value, default in zip(
            custom_normal_allocation.values(), normal_allocation.values()
        )) else (
            anomaly_allocation if row['predicted_crisis'] == 1 else normal_allocation
        )

        # Calculate returns
        market_data = {
            'stocks': sum(row[stock] for stock in stocks),
            'bonds': sum(row[bond] for bond in bonds),
            'cash': 1.0
        }

        returns = {
            'stocks': ((market_data['stocks'] / prev_row['stocks'] - 1) if prev_row else 0),
            'bonds': ((market_data['bonds'] / prev_row['bonds'] - 1) if prev_row else 0),
            'cash': 0.0006
        } if prev_row else {'stocks': 0, 'bonds': 0, 'cash': 0}

        # Update portfolio value
        portfolio_value += sum(
            allocation[asset] / 100 * portfolio_value * returns[asset]
            for asset in returns
        )

        portfolio_history.append((row['Data'], portfolio_value))
        prev_row = market_data

    return portfolio_value, portfolio_history

# Run portfolio simulation
final_value, portfolio_history = simulate_strategy(
    filtered_data, selected_stocks, selected_bonds, starting_value
)

# Portfolio results
st.title("Portfolio Simulation and Analysis")
st.write(f"Final Portfolio Value: ${final_value:,.2f}")

# Convert portfolio history to a DataFrame
portfolio_df = pd.DataFrame(portfolio_history, columns=["Data", "Portfolio Value"])
portfolio_df.set_index("Data", inplace=True)

# Visualization
st.line_chart(portfolio_df["Portfolio Value"])

# Set up GROQ API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_EVAL_MODEL = "llama-3.3-70b-versatile"

# Title and Description
st.header("Market Analysis")

# Initialize SHAP explainer
explainer = shap.Explainer(model, X_new)
shap_values = explainer(X_new)

# Map dates to indices
date_to_index = {date: idx for idx, date in enumerate(filtered_data['Data'])}
index_to_date = {idx: date for date, idx in date_to_index.items()}

# Add date-based selection
selected_date = st.selectbox("Select Date for SHAP Analysis", options=filtered_data['Data'])
selected_index = date_to_index[selected_date]

st.subheader(f"SHAP Analysis For: {selected_date}")

# Generate SHAP waterfall plot for the selected date
fig, ax = plt.subplots(figsize=(10, 5))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[selected_index].values,
        base_values=shap_values[selected_index].base_values,
        feature_names=X_new.columns.tolist()
    ),
    show=False
)
plt.tight_layout()  # Adjust layout for Streamlit
st.pyplot(fig)

# Prepare SHAP explanation for the LLM
shap_details = {
    "date": str(selected_date),
    "base_value": shap_values[selected_index].base_values.tolist(),
    "feature_values": dict(zip(X_new.columns, X_new.iloc[selected_index].tolist())),
    "shap_values": dict(zip(X_new.columns, shap_values[selected_index].values.tolist()))
}

# Pass SHAP details to GROQ's LLM for explanation
llm_input = f"""
Explain concisely why the model predicted that {selected_date} is an {'anomaly' if predictions[selected_index] else 'normal'} day. 
Use the SHAP values to explain how each factor influenced the prediction and clarify what the numbers mean in this context. 
Here’s the breakdown:
- The Base Value ({shap_details['base_value']}) is the model’s prediction for an average day with no additional information.
- Feature Contributions ({shap_details['shap_values']}) are the amounts each factor added to or subtracted from the Base Value to reach the final prediction. Positive values push toward an anomaly prediction, and negative values push toward a normal day.
- Feature Values ({shap_details['feature_values']}) are the actual values of these factors on {selected_date}, compared to what the model expects for an average day.

Explain how the contributions and actual factor values influenced the prediction in the context of whether today is an anomaly or normal, without using technical jargon or unrelated details.
"""

# Send request to GROQ's API
try:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_EVAL_MODEL,
        "messages": [
            {"role": "system", "content": "You are an evaluator. Explain the model prediction based on the SHAP details."},
            {"role": "user", "content": llm_input}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        llm_response = response.json()["choices"][0]["message"]["content"]
        st.subheader("Explanation")
        st.write(llm_response)
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
except Exception as e:
    st.error(f"An error occurred while fetching the LLM response: {e}")

# Start the Streamlit app
if __name__ == "__main__":
    st.run()

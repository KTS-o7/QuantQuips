# app.py
import streamlit as st
import subprocess

def save_strategy(strategy_code):
    with open('strategies.py', 'w') as file:
        file.write(strategy_code)

def run_trader(principal_amount):
    try:
        result = subprocess.run(['python', 'trader.py', str(principal_amount)], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr

# Streamlit app
st.title("Automated Trader App")

# Text area for user input (strategy code)
strategy_code = st.text_area("Enter your strategy code here:")

# Input for initial principal amount
principal_amount = st.number_input("Enter initial principal amount:", value=1000000000.0)

# Button to save the strategy and run the trader
if st.button("Save and Run Trader"):
    save_strategy(strategy_code)
    st.write("Strategy saved successfully!")
    
    # Run trader.py with initial principal amount as argument
    output = run_trader(principal_amount)
    
    # Display the output
    st.subheader("Trader Output:")
    st.text(output)

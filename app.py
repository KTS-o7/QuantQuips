from __future__ import (absolute_import, division, print_function,unicode_literals)
import streamlit as st
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta
from io import StringIO
import contextlib
import textwrap
import backtrader as bt
import sys  # Library for system-specific parameters and functions
import pandas as pd
from gpt4all import GPT4All
from pathlib import Path
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import subprocess

# Function to load documents 
def load_documents():
    loader = DirectoryLoader('data/data', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Function to split text into chunks
def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Function to create embeddings
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cuda"})
    return embeddings

# Function to create vector store
def create_vector_store(text_chunks, embeddings):
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store

# Function to create LLMS model
def create_llms_model():
    llm = CTransformers(model="mistral-7b-openorca.Q4_0.gguf", config={'max_new_tokens': 512, 'temperature': 0.01,'gpu_layers': 10})
    return llm
documents = load_documents()

        # Split text into chunks
text_chunks = split_text_into_chunks(documents)




st.set_page_config(page_title="QuantQuips", page_icon="chart_with_upwards_trend", layout='wide')
# Set page title
st.title("QuantQuips")
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)



# Function to fetch real-time stock data during market hours
def fetch_realtime_stock_data(ticker_symbol, period, interval):
    current_time = datetime.now().time()
    
    try:
            stock_data = yf.download(ticker_symbol, period=period, interval=interval)
    except Exception as e:
            # Retry with a different interval if the initial request fails
            if "15m data not available" in str(e):
                st.warning(f"15-minute data not available for the specified period. Fetching hourly data instead.")
                stock_data = yf.download(ticker_symbol, period=period, interval="1h")
            else:
                raise
    return stock_data
    

# Default values for period and interval
default_period = "1d"
default_interval = "1m"

# Fetch and display real-time stock data for NSE
nse_ticker_symbol = "^NSEI"
nse_stock_data = fetch_realtime_stock_data(nse_ticker_symbol, default_period, default_interval)

# Fetch and display real-time stock data for Sensex
sensex_ticker_symbol = "^BSESN"
sensex_stock_data = fetch_realtime_stock_data(sensex_ticker_symbol, default_period, default_interval)

# Calculate the percentage change for NSE
nse_percentage_change = (nse_stock_data['Close'].iloc[-1] - nse_stock_data['Close'].iloc[0]) / nse_stock_data['Close'].iloc[0] * 100

# Calculate the percentage change for Sensex
sensex_percentage_change = (sensex_stock_data['Close'].iloc[-1] - sensex_stock_data['Close'].iloc[0]) / sensex_stock_data['Close'].iloc[0] * 100

# Determine overall market condition
overall_market_condition = 'Bullish' if nse_percentage_change > 0 and sensex_percentage_change > 0 else 'Bearish'

# Sidebar for navigation
# Add a title and a selectbox to the sidebar
st.sidebar.markdown("## **Navigation**")
page_selection = st.sidebar.radio("**Goto**", ["Home", "Backtesting", "Genetic Algorithm","LLM","About Us"],captions=["Goes to home page","Goes to backtesting page","Goes to genetic algorithm","Goes to LLM","Goes to about us page"])


# Default page
def plot_chart(stock_data, title, subheader):
    fig = px.line(stock_data, x=stock_data.index, y="Close", title=title)
    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='Closing Price')
    st.subheader(subheader)
    if overall_market_condition == 'Bullish':
        fig.update_traces(line_color='light-blue')
    else:
        fig.update_traces(line_color='red')
    st.plotly_chart(fig, use_container_width=True, width=1200)
    st.subheader(f"Latest Data for {title}")
    st.write(stock_data.tail())  # Display the last rows of the DataFrame

if page_selection == "Home":
    # Display overall market condition and real-time stock charts
    st.subheader(f"Overall Market Condition: {overall_market_condition}")

    # Display NSE and Sensex charts side by side with larger width
    if nse_stock_data is not None and sensex_stock_data is not None:
        col1, col2 = st.columns(2)

        with col1:
            plot_chart(nse_stock_data, "Real-time Nifty Chart", "Real-time Nifty Chart:")

        with col2:
            plot_chart(sensex_stock_data, "Real-time Sensex Chart", "Real-time Sensex Chart:")
    else:
        st.warning("Nifty or Sensex market is closed. Real-time data is available only during market hours.")
    st.write("Auto-refreshing every 1 minute.")
























elif page_selection == "Genetic Algorithm":


        # Streamlit code
    st.title('Genetic Algorithm')

    # Text area for user to enter Python code
    st.header("Enter Your Trading Strategy")
    strategy_code = st.text_area('Enter your Python code:', value='', height=None, max_chars=None, key=None)
    
    

    st.header("Parameter Ranges")
    short_period_range = st.slider("Short Period Range", min_value=10, max_value=50, value=(15, 30))
    long_period_range = st.slider("Long Period Range", min_value=40, max_value=100, value=(40, 60))
    # Input for initial cash
    cash = st.number_input('Enter initial cash:', value=10000.0)
    if (cash < 0.0):
        st.warning('Initial cash must be a positive number')
        

    # Input for data file
    data_file = st.text_input('Enter path to data file:', value='')
    
    # Button to run backtest
    if st.button('Run Backtest'):
        # Redirect stdout to a string buffer
        buf = StringIO()
        sys.stdout = buf

        # Wrap code in a function and add necessary imports
        code = textwrap.dedent(f"""
import backtrader as bt
import datetime
import random
import numpy as np
best = []
{strategy_code}

                
def run_strategy(short_period, long_period):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MovingAverageCrossoverStrategy, short_period=short_period, long_period=long_period)
    data = bt.feeds.YahooFinanceCSVData(dataname='{data_file}', fromdate=datetime.datetime(2021, 1, 1), todate=datetime.datetime(2022, 1, 1))
    cerebro.adddata(data)
    cerebro.broker.set_cash({cash})
    cerebro.broker.setcommission(commission=0.001)
    cerebro.run()

    #print(cerebro.broker.getvalue())
    return cerebro.broker.getvalue()  # or some other performance metric

def select_best(results):
    # Sort the results by performance
    max_tuple = max(results, key=lambda x: x[2])
    #print(max_tuple)
    best.append(max_tuple)
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)

    # Select the top half of the population
    return [x[:2] for x in sorted_results[:int(POPULATION_SIZE // 2)]]

def crossover_and_mutate(population, mutation_rate):
    next_generation = []
    for _ in range(int(POPULATION_SIZE)):
                # Select two parents
        parent1, parent2 = random.sample(population, 2)
                # Perform crossover
        child = (parent1[0], parent2[1]) if random.random() < 0.5 else (parent2[0], parent1[1])
                # Perform mutation
        child = (int(child[0] + np.random.normal(0, mutation_rate)), int(child[1] + np.random.normal(0, mutation_rate)))
        next_generation.append(child)
    return next_generation

        # Define the genetic algorithm parameters
POPULATION_SIZE = 10
GENERATIONS = 50
MUTATION_RATE = 0.05
        # Generate the initial population
population = [(random.randint(15, 30), random.randint(40, 60)) for _ in range(int(POPULATION_SIZE))]

for _ in range(int(GENERATIONS)):
            # Evaluate the population
    results = [(short_period, long_period, run_strategy(short_period, long_period)) for short_period, long_period in population]
            # Select the best individuals for the next generation
    population = select_best(results)
            # Apply crossover and mutation to generate the next generation
    population = crossover_and_mutate(population, MUTATION_RATE)
max_tuple2 = max(best, key=lambda x: x[2])

print(max_tuple2[0],max_tuple2[1],max_tuple2[2])
        """)

        # Execute the code
        exec(code)

        # Get stdout contents
        output = buf.getvalue()

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Display output
        outputList = output.split()
        shortprd = int(outputList[0])
        longprd = int(outputList[1])
        result = float(outputList[2])/10
        profit = round(result - cash,2)
        result = round(result,2)
        
        
        
        st.markdown("""
    <div style="color: #ffffff; font-size: 24px; font-weight: bold;">Recommendation</div>
    <ul>
        <li style="color: #ffffff; font-size: 20px;">Short period : {shortprd}</li>
        <li style="color: #ffffff; font-size: 20px;">Large period : {largeprd}</li>
        <li style="color: #ffffff; font-size: 20px;">Portfolio value : {result}</span> 
        <span style="font-weight: bold;">Profit : {profit}</span>.</li>
    </ul>
    """, unsafe_allow_html=True)
       
        
## LLM PART
elif page_selection == "LLM":
    st.title("QuantQuip-ChatBot")
   
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader('Get your finances up and above 💪')
    st.markdown('<style>h3{color: pink; text-align: center;}</style>', unsafe_allow_html=True)

    

        # Create embeddings
    embeddings = create_embeddings()

        # Create vector store
    vector_store = create_vector_store(text_chunks, embeddings) 
    # Create LLMS model
    llm = create_llms_model()
   

        # Initialize conversation history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about finance 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    # Create memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                memory=memory)

    # Define chat function
    def conversation_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    # Display chat history
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your portfolio", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
## ABOUT US

elif page_selection == "About Us":
    st.title("About Us")
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="color: #ffffff; font-size: 24px; font-weight: bold;">Welcome to our stock analysis and backtesting platform.</div>
    <ul>
        <li style="color: #ffffff; font-size: 20px;">We are a group of passionate undergraduate students who are on a mission to make life easier for people dipping their toes into the fintech world.</li>
        <li style="color: #ffffff; font-size: 20px;">Our platform provides users with the tools they need to backtest genetic algorithms and carry out successful trades.</li>
        <li style="color: #ffffff; font-size: 20px;">With our innovative approach, we aim to revolutionize the <span style="font-weight: bold;">way people interact with financial markets</span> and <span style="font-weight: bold;">empower them to make informed decisions</span>.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown('<style>h3{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader("Our Team")
    st.markdown("""<div style="color: white; font-size: 24px; font-weight: bold;">Meet our team of talented individuals who are dedicated to making a difference in the fintech world.</div>
    """,unsafe_allow_html=True)

    col1, col2, col3 ,col4 = st.columns(4)
    with col1:
        st.image("https://avatars.githubusercontent.com/u/45279662?v=4", use_column_width=True)
        st.markdown("""
        #### Krishnatejaswi S
        Langchain Developer
        """)
    with col2:
        st.image("https://avatars.githubusercontent.com/u/45279662?v=4", use_column_width=True)
        st.markdown("""
        #### Vinayak C
        ML Engineer
        """)
    with col3:
        st.image("https://avatars.githubusercontent.com/u/45279662?v=4", use_column_width=True)
        st.markdown("""
        #### Bipin Raj C
        Python Developer
        """)
    with col4:
        st.image("https://avatars.githubusercontent.com/u/45279662?v=4", use_column_width=True)
        st.markdown("""
        #### Ananya Bhat
        Python Developer
        """)

elif page_selection == "Backtesting":
    def save_strategy(strategy_code):
        with open('strategies.py', 'w') as file:
                file.write(strategy_code)

    def run_trader(principal_amount):
        try:
            result = subprocess.run(['python', 'trader.py', str(principal_amount)], capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return e.stderr
    st.title("Automated Back Testing")

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



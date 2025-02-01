import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import logging
from crewai import Agent, Task, Crew, Process
from openai import OpenAI
from statsmodels.tsa.arima.model import ARIMA
import yaml

# ---------------------------
# Environment & Clients Setup
# ---------------------------
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
openai_api_key = os.getenv("OPENAI_API_KEY")

client_db = MongoClient(mongo_uri)
db = client_db['crude_oil_analysis']
prices_collection = db['prices']
news_collection = db['news']
client = OpenAI(api_key=openai_api_key)

# ---------------------------
# YAML Configuration Loader
# ---------------------------
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

agents_config = load_yaml_config('config/agents.yaml')
tasks_config = load_yaml_config('config/task.yaml')

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Helper Functions
# ---------------------------
def fetch_crude_oil_data():
    # Fetch only the crude oil trend (assumed to be "Dated Brent Oil Price")
    data = list(prices_collection.find({"trend": "Dated Brent Oil Price"}).sort("date", -1).limit(30))
    if data:
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()  # ascending order
        return df[["Close"]].tail(14)
    else:
        logger.warning("No crude oil price data available in MongoDB")
        return None

def forecast_crude_oil(data):
    data_length = len(data)
    if data_length >= 30:
        order = (10, 2, 5)
    elif data_length >= 15:
        order = (5, 1, 3)
    elif data_length >= 5:
        order = (2, 1, 1)
    else:
        logger.warning("Insufficient data for ARIMA forecasting.")
        return pd.Series()
    try:
        model = ARIMA(data["Close"], order=order)
        model_fit = model.fit()
        return model_fit.forecast(steps=7)
    except Exception as e:
        logger.error(f"Error during ARIMA forecasting: {e}")
        return pd.Series()

def forecast_series(series):
    """Generic forecasting for a pandas Series using ARIMA."""
    data_length = len(series)
    if data_length >= 30:
        order = (10, 2, 5)
    elif data_length >= 15:
        order = (5, 1, 3)
    elif data_length >= 5:
        order = (2, 1, 1)
    else:
        logger.warning("Insufficient data for ARIMA forecasting.")
        return pd.Series()
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        return model_fit.forecast(steps=7)
    except Exception as e:
        logger.error(f"Error during ARIMA forecasting: {e}")
        return pd.Series()

def fetch_news():
    articles = list(news_collection.find().sort("timestamp", -1).limit(7))
    if articles:
        return "\n\n".join([f"{article['headline']}: {article['story']}" for article in articles])
    else:
        logger.warning("No news data available in MongoDB")
        return "No news data available."

# ---------------------------
# Agent Classes for Crude Oil
# ---------------------------
class AnalysisAgent(Agent):
    def __init__(self):
        config = agents_config['analysis_agent']
        super().__init__(role=config['role'], goal=config['goal'], backstory=config['backstory'], allow_delegation=True)

    def run(self, data, forecast, news_summary):
        if data is None or data.empty:
            logger.warning("Insufficient data for analysis.")
            return "Data is insufficient to provide an analysis."

        if forecast is None or forecast.empty:
            forecast_summary = "Forecast data is unavailable."
        else:
            trend = "upward" if data["Close"].iloc[-1] > data["Close"].iloc[0] else "downward"
            forecast_trend = "increasing" if forecast.iloc[-1] > data["Close"].iloc[-1] else "decreasing"
            forecast_summary = (
                f"The historical crude oil price trend shows a {trend} movement. "
                f"The 7-day forecast indicates prices will be {forecast_trend}. "
                f"Latest price: ${data['Close'].iloc[-1]:.2f}, forecasted: ${forecast.iloc[-1]:.2f}."
            )

        prompt = f"{forecast_summary}\n\nRelevant News:\n{news_summary if news_summary else 'No recent news available.'}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing crude oil market analysis. Provide a concise analysis in 4-5 paragraphs."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content

class ChatAgent(Agent):
    def __init__(self):
        config = agents_config['chat_agent']
        super().__init__(role=config['role'], goal=config['goal'], backstory=config['backstory'], allow_delegation=True)

    def run(self, user_input, analysis_summary):
        system_message = (
            "You are a knowledgeable assistant providing insights on crude oil prices. "
            "Base your responses on the provided analysis summary. Answer concisely in up to 3 paragraphs."
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "assistant", "content": analysis_summary},
                {"role": "user", "content": user_input}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content

# ---------------------------
# Agent Classes for Petroleum Products
# ---------------------------
class AnalysisAgentPetroleum(Agent):
    def __init__(self):
        # You may reuse the same config or adjust as needed
        config = agents_config['analysis_agent']
        super().__init__(role=config['role'], goal="Provide analysis for petroleum product crack spreads.",
                         backstory=config['backstory'], allow_delegation=True)

    def run(self, data_dict, forecast_dict, news_summary):
        summary = ""
        for trend, df in data_dict.items():
            if df is None or df.empty:
                summary += f"For {trend}, data is insufficient.\n"
            else:
                historical = df["Close"]
                forecast_series_ = forecast_dict.get(trend, pd.Series())
                if forecast_series_.empty:
                    forecast_info = "Forecast data is unavailable."
                else:
                    trend_direction = "upward" if historical.iloc[-1] > historical.iloc[0] else "downward"
                    forecast_trend = "increasing" if forecast_series_.iloc[-1] > historical.iloc[-1] else "decreasing"
                    forecast_info = (
                        f"The historical trend is {trend_direction} and the 7-day forecast suggests prices will be {forecast_trend}. "
                        f"Latest price: ${historical.iloc[-1]:.2f}, forecasted: ${forecast_series_.iloc[-1]:.2f}."
                    )
                summary += f"{trend}: {forecast_info}\n"

        prompt = f"Analysis for petroleum product crack spreads:\n{summary}\n\nRelevant News:\n{news_summary if news_summary else 'No recent news available.'}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing analysis on petroleum product crack spreads. Provide a concise analysis in 4-5 paragraphs."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content

class ChatAgentPetroleum(Agent):
    def __init__(self):
        config = agents_config['chat_agent']
        super().__init__(role=config['role'], goal="Answer queries on petroleum product crack spreads.",
                         backstory=config['backstory'], allow_delegation=True)

    def run(self, user_input, analysis_summary):
        system_message = (
            "You are a knowledgeable assistant providing insights on petroleum product crack spreads. "
            "Base your answers on the provided analysis summary. Answer concisely in up to 3 paragraphs."
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "assistant", "content": analysis_summary},
                {"role": "user", "content": user_input}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content

# ---------------------------
# Crew Setup Functions
# ---------------------------
def setup_crew_crude():
    data = fetch_crude_oil_data()
    forecast = forecast_crude_oil(data) if data is not None else None
    news_summary = fetch_news()

    if data is None or forecast is None or news_summary is None:
        logger.error("Failed to retrieve all necessary data for crude oil crew setup.")
        return None, None, None, None, None

    analysis_agent = AnalysisAgent()
    chat_agent = ChatAgent()
    analysis_summary = analysis_agent.run(data, forecast, news_summary)
    return (analysis_summary, data, forecast, news_summary, (analysis_agent, chat_agent))

def setup_crew_petroleum():
    trends = [
        "Naphtha Crack Spread",
        "Gasoline 95 Crack Spread",
        "Gasoline 97 Crack Spread",
        "Gasoil 10PPM Crack Spread",
        "Gasoil 500PPM Crack Spread",
        "Gasoil 2500PPM Crack Spread"
    ]
    data_dict = {}
    forecast_dict = {}
    news_summary = fetch_news()

    for trend in trends:
        trend_data = list(prices_collection.find({"trend": trend}).sort("date", -1).limit(30))
        if trend_data:
            df = pd.DataFrame(trend_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df = df[["Close"]].tail(14)
            data_dict[trend] = df
            forecast_dict[trend] = forecast_series(df["Close"])
        else:
            logger.warning(f"No data for trend: {trend}")
            data_dict[trend] = pd.DataFrame()
            forecast_dict[trend] = pd.Series()

    analysis_agent = AnalysisAgentPetroleum()
    chat_agent = ChatAgentPetroleum()
    analysis_summary = analysis_agent.run(data_dict, forecast_dict, news_summary)
    return (analysis_summary, data_dict, forecast_dict, news_summary, (analysis_agent, chat_agent))

# ---------------------------
# Streamlit App Layout & Navigation
# ---------------------------
st.set_page_config(layout="wide")
st.title("Market Analysis by MEGAT AI Agents")

# Sidebar Navigation: Choose between the two pages
page = st.sidebar.radio("Select Analysis Page", ["Crude Oil Analysis", "Petroleum Products Analysis"])

# ---------------------------
# Page 1: Crude Oil Analysis
# ---------------------------
if page == "Crude Oil Analysis":
    st.header("Crude Oil Prices")
    crew_result = setup_crew_crude()
    if crew_result[0] is None:
        st.error("Failed to initialize Crude Oil Analysis. Please check your API keys and network connectivity.")
    else:
        analysis_summary, data, forecast, news_summary, agents = crew_result
        col1, col2 = st.columns([1, 2])
        with col2:
            st.subheader("Market Analysis")
            if data is not None and forecast is not None:
                historical_data = data["Close"].to_list()
                future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=len(forecast))
                forecast_data = forecast.to_list()

                # Combine historical and forecast data
                chart_data = pd.DataFrame({
                    "Date": data.index.tolist() + future_dates.tolist(),
                    "Close": historical_data + forecast_data,
                    "Type": ["Historical"] * len(historical_data) + ["Forecast"] * len(forecast_data)
                }).set_index("Date")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chart_data[chart_data['Type'] == 'Historical'].index,
                    y=chart_data[chart_data['Type'] == 'Historical']['Close'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=chart_data[chart_data['Type'] == 'Forecast'].index,
                    y=chart_data[chart_data['Type'] == 'Forecast']['Close'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red')
                ))
                fig.update_layout(
                    title="Historical and Forecasted Crude Oil Prices",
                    xaxis_title="Date",
                    yaxis_title="Close Price",
                    hovermode="x unified",
                    width=1000,
                    height=600
                )
                st.plotly_chart(fig)
                st.write(analysis_summary)
            else:
                st.warning("Unable to display chart due to missing data.")
        with col1:
            st.subheader("Ask the AI about Crude Oil Trends")
            if "messages_crude" not in st.session_state:
                st.session_state.messages_crude = []
            for message in st.session_state.messages_crude:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            if prompt := st.chat_input("Your question:"):
                st.session_state.messages_crude.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                chat_agent = agents[1]  # ChatAgent for crude oil
                full_response = chat_agent.run(prompt, analysis_summary)
                with st.chat_message("assistant"):
                    st.markdown(full_response)
                st.session_state.messages_crude.append({"role": "assistant", "content": full_response})

# ---------------------------
# Page 2: Petroleum Products Analysis
# ---------------------------
else:
    st.header("Petroleum Products Crack Spreads")
    crew_result = setup_crew_petroleum()
    if crew_result[0] is None:
        st.error("Failed to initialize Petroleum Products Analysis. Please check your API keys and network connectivity.")
    else:
        analysis_summary, data_dict, forecast_dict, news_summary, agents = crew_result
        st.subheader("Market Analysis")
        fig = go.Figure()
        # For each petroleum product trend, add historical and forecast traces to the same chart.
        for trend, df in data_dict.items():
            if df is not None and not df.empty and trend in forecast_dict and not forecast_dict[trend].empty:
                historical = df["Close"].to_list()
                future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=len(forecast_dict[trend]))
                forecast_vals = forecast_dict[trend].to_list()

                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=historical,
                    mode='lines',
                    name=f'{trend} Historical'
                ))
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast_vals,
                    mode='lines',
                    name=f'{trend} Forecast',
                    line=dict(dash='dash')
                ))
            else:
                st.warning(f"Insufficient data for {trend}.")
        fig.update_layout(
            title="Historical and Forecasted Petroleum Product Crack Spreads",
            xaxis_title="Date",
            yaxis_title="Close Price",
            hovermode="x unified",
            width=1000,
            height=600
        )
        st.plotly_chart(fig)
        st.write(analysis_summary)
        st.subheader("Ask the AI about Petroleum Product Trends")
        if "messages_petroleum" not in st.session_state:
            st.session_state.messages_petroleum = []
        for message in st.session_state.messages_petroleum:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("Your question:"):
            st.session_state.messages_petroleum.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            chat_agent = agents[1]  # ChatAgent for petroleum products
            full_response = chat_agent.run(prompt, analysis_summary)
            with st.chat_message("assistant"):
                st.markdown(full_response)
            st.session_state.messages_petroleum.append({"role": "assistant", "content": full_response})

# crew_setup.py

import os
import logging
import requests
import pandas as pd
from datetime import timedelta
from dotenv import load_dotenv
import yaml
from crewai import Agent, Task, Crew, Process
from openai import OpenAI
from statsmodels.tsa.arima.model import ARIMA

# Load environment variables
load_dotenv()
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YAML configuration files
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

agents_config = load_yaml_config('config/agents.yaml')
tasks_config = load_yaml_config('config/task.yaml')

# Fetch crude oil data from Alpha Vantage API
def fetch_crude_oil_data():
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=CL&apikey={alpha_vantage_api_key}&outputsize=compact"
    response = requests.get(url)
    data = response.json()
    if "Time Series (Daily)" in data:
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.rename(columns={"4. close": "Close"})
        return df[["Close"]].tail(14)
    else:
        logger.warning("Failed to fetch crude oil data")
        return None

# Forecast crude oil prices using ARIMA
def forecast_crude_oil(data):
    try:
        model = ARIMA(data["Close"], order=(10, 2, 5))
        model_fit = model.fit()
        return model_fit.forecast(steps=7)
    except Exception as e:
        logger.error(f"Error during ARIMA forecasting: {e}")
        return pd.Series()

# Fetch relevant news articles from News API
def fetch_news():
    keywords = ["crude oil", "OPEC", "geopolitical"]
    query = " OR ".join(keywords)
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize=7&sortBy=publishedAt&language=en&apiKey={news_api_key}"
    response = requests.get(url)
    news_data = response.json()
    if news_data.get("status") == "ok":
        articles = news_data["articles"]
        return "\n\n".join([f"{article['title']}: {article['description']}" for article in articles[:7]])
    else:
        logger.warning("Failed to fetch news data")
        return "No news data available."

# Define AnalysisAgent and ChatAgent using YAML configuration
class AnalysisAgent(Agent):
    def __init__(self):
        config = agents_config['analysis_agent']
        super().__init__(role=config['role'], goal=config['goal'], backstory=config['backstory'], allow_delegation=True)

    def run(self, data, forecast, news_summary):
        trend = "upward" if data["Close"].iloc[-1] > data["Close"].iloc[0] else "downward"
        forecast_trend = "increasing" if forecast.iloc[-1] > data["Close"].iloc[-1] else "decreasing"
        
        prompt = (
            f"The current crude oil price trend over the past two weeks shows a {trend} movement. "
            f"The 7-day forecast indicates prices are likely to be {forecast_trend}. "
            f"The latest price is ${data['Close'].iloc[-1]:.2f}, with a forecasted price of ${forecast.iloc[-1]:.2f}.\n\n"
            f"Relevant News:\n{news_summary}"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing crude oil market analysis."},
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

    # In ChatAgent
    def run(self, user_input, analysis_summary):
        # System message to instruct the AI to use the provided analysis summary
        system_message = (
            "You are a knowledgeable assistant providing insights on crude oil prices. "
            "You have been provided with the latest analysis summary, which includes current crude oil price trends, "
            "forecasted prices, and relevant news. Please base your responses on this information and answer questions "
            "as if you have real-time data."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "assistant", "content": analysis_summary},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content


# Set up Crew with agents and tasks using YAML configuration
def setup_crew():
    data = fetch_crude_oil_data()
    forecast = forecast_crude_oil(data) if data is not None else None
    news_summary = fetch_news()

    if data is None or forecast is None or news_summary is None:
        logger.error("Failed to retrieve all necessary data for crew setup.")
        return None, None, None, None  # Return None if data fetching fails

    analysis_agent = AnalysisAgent()
    chat_agent = ChatAgent()

    analysis_task = Task(
        description=tasks_config['crude_oil_analysis']['description'],
        agent=analysis_agent,
        expected_output=tasks_config['crude_oil_analysis']['expected_output']
    )

    chat_task = Task(
        description=tasks_config['recommendation']['description'],
        agent=chat_agent,
        expected_output=tasks_config['recommendation']['expected_output'],
        context=[analysis_task]
    )

    crew = Crew(
        agents=[analysis_agent, chat_agent],
        tasks=[analysis_task, chat_task],
        process=Process.sequential
    )

    analysis_summary = analysis_agent.run(data, forecast, news_summary)
    return crew, analysis_summary, data, forecast, news_summary


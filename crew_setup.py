import os
import logging
import pandas as pd
from datetime import timedelta
from dotenv import load_dotenv
import yaml
from pymongo import MongoClient
from crewai import Agent, Task, Crew, Process
from openai import OpenAI
from statsmodels.tsa.arima.model import ARIMA

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")  # MongoDB URI
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize MongoDB client and OpenAI client
client_db = MongoClient(mongo_uri)
db = client_db['crude_oil_analysis']
prices_collection = db['prices']
news_collection = db['news']
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

# Fetch the last 30 days of crude oil price data from MongoDB
def fetch_crude_oil_data():
    data = list(prices_collection.find().sort("date", -1).limit(30))
    if data:
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()  # Ensure data is sorted by date ascending
        return df[["Close"]].tail(14)
    else:
        logger.warning("No crude oil price data available in MongoDB")
        return None

# Forecast crude oil prices using ARIMA
def forecast_crude_oil(data):
    # Check if there are enough data points for a robust ARIMA model
    data_length = len(data)
    
    # Dynamically adjust the ARIMA model order based on the data length
    if data_length >= 30:
        order = (10, 2, 5)  # Optimal order for larger datasets
    elif data_length >= 15:
        order = (5, 1, 3)  # Adjusted order for moderate data availability
    elif data_length >= 5:
        order = (2, 1, 1)  # Basic model for limited data
    else:
        logger.warning("Insufficient data for ARIMA forecasting.")
        return pd.Series()  # Return an empty Series if data is too sparse
    
    try:
        model = ARIMA(data["Close"], order=order)
        model_fit = model.fit()
        return model_fit.forecast(steps=7)  # Forecasting the next 7 days
    except Exception as e:
        logger.error(f"Error during ARIMA forecasting: {e}")
        return pd.Series()


# Fetch the latest news articles from MongoDB
def fetch_news():
    articles = list(news_collection.find().sort("publishedAt", -1).limit(7))
    if articles:
        return "\n\n".join([f"{article['headline']}: {article['story']}" for article in articles])
    else:
        logger.warning("No news data available in MongoDB")
        return "No news data available."

class AnalysisAgent(Agent):
    def __init__(self):
        config = agents_config['analysis_agent']
        super().__init__(role=config['role'], goal=config['goal'], backstory=config['backstory'], allow_delegation=True)

    def run(self, data, forecast, news_summary):
        # Check if data and forecast are available and not empty
        if data is None or data.empty:
            logger.warning("Insufficient data for analysis.")
            return "Data is insufficient to provide an analysis."

        # Handle forecast availability
        if forecast is None or forecast.empty:
            logger.warning("Forecast data is unavailable.")
            forecast_summary = "Forecast data is unavailable."
        else:
            # Determine recent trend and forecast trend based on available data
            trend = "upward" if data["Close"].iloc[-1] > data["Close"].iloc[0] else "downward"
            forecast_trend = (
                "increasing" if forecast.iloc[-1] > data["Close"].iloc[-1] else "decreasing"
            )
            forecast_summary = (
                f"The current crude oil price trend over the past period shows a {trend} movement. "
                f"The 7-day forecast indicates prices are likely to be {forecast_trend}. "
                f"The latest price is ${data['Close'].iloc[-1]:.2f}, with a forecasted price of ${forecast.iloc[-1]:.2f}."
            )

        # Build the analysis summary    
        prompt = (
            f"{forecast_summary}\n\n"
            f"Relevant News:\n{news_summary if news_summary else 'No recent news data available.'}"
        )

        # Generate the response using OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing crude oil market analysis. Your analysis should be concise and within 4-5 paragraphs"},
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
            "as if you have real-time data. Keep your answers conside within 3 paragraphs"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "assistant", "content": analysis_summary},
                {"role": "user", "content": user_input}
            ],
            max_tokens=300,
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
        return None, None, None, None, None  # Return None if data fetching fails

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

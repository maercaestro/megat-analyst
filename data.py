import os
import logging
import requests
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")
mongo_uri = os.getenv("MONGO_URI")  # MongoDB URI

# Initialize MongoDB client
client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
db = client['crude_oil_analysis']
prices_collection = db['prices']
news_collection = db['news']

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch and store the last 14 days of crude oil price data in MongoDB
def fetch_and_store_crude_oil_data():
    # Delete old data from the prices collection
    prices_collection.delete_many({})  # Clears all previous price data

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=CL&apikey={alpha_vantage_api_key}&outputsize=compact"
    response = requests.get(url)
    data = response.json()
    
    if "Time Series (Daily)" in data:
        # Create a DataFrame and sort it by date
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.rename(columns={"4. close": "Close"})
        
        # Get the last 14 days of data
        last_14_days_data = df.tail(14)

        # Insert each of the last 14 days into MongoDB
        for date, row in last_14_days_data.iterrows():
            prices_collection.insert_one({
                "date": date,
                "Close": row['Close']
            })
        logger.info("Last 14 days of crude oil price data stored in MongoDB")
    else:
        logger.warning("Failed to fetch crude oil data")

# Fetch and store relevant news articles in MongoDB
def fetch_and_store_news_data():
    # Delete old data from the news collection
    news_collection.delete_many({})  # Clears all previous news data

    keywords = ["crude oil", "OPEC", "geopolitical"]
    query = " OR ".join(keywords)
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize=7&sortBy=publishedAt&language=en&apiKey={news_api_key}"
    response = requests.get(url)
    news_data = response.json()
    
    if news_data.get("status") == "ok":
        articles = news_data["articles"]
        
        # Insert the latest news articles into MongoDB
        for article in articles:
            news_entry = {
                "title": article["title"],
                "description": article["description"],
                "publishedAt": datetime.strptime(article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"),
                "source": article["source"]["name"],
                "url": article["url"]
            }
            news_collection.insert_one(news_entry)
        logger.info("Latest news data stored in MongoDB")
    else:
        logger.warning("Failed to fetch news data")

if __name__ == "__main__":
    # Fetch and store data in MongoDB
    fetch_and_store_crude_oil_data()
    fetch_and_store_news_data()
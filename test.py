import os
import logging
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")  # MongoDB URI

# Initialize MongoDB client
client_db = MongoClient(mongo_uri)
db = client_db['crude_oil_analysis']
prices_collection = db['prices']
news_collection = db['news']

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test fetching crude oil price data
def test_fetch_crude_oil_data():
    data = list(prices_collection.find().sort("date", -1).limit(30))
    if data:
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        print("Crude Oil Price Data (Last 30 Days):")
        print(df[["Close"]])
    else:
        print("No crude oil price data available in MongoDB.")

# Test fetching news data
def test_fetch_news_data():
    articles = list(news_collection.find().sort("publishedAt", -1).limit(7))
    if articles:
        print("\nNews Data (Latest 7 Articles):")
        for article in articles:
            print(f"Title: {article['title']}")
            print(f"Description: {article['description']}")
            print(f"Published At: {article['publishedAt']}\n")
    else:
        print("No news data available in MongoDB.")

if __name__ == "__main__":
    print("Testing Data Fetching from MongoDB...\n")
    test_fetch_crude_oil_data()
    test_fetch_news_data()

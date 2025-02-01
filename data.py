import os
import logging
import requests
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")  # MongoDB URI
enverus_api_key = os.getenv("ENVERUS_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)  # Reduced verbosity to INFO level
logger = logging.getLogger(__name__)

# MongoDB setup
client = MongoClient(mongo_uri)
db = client['crude_oil_analysis']  # Database name

# Fetch data from Enverus API
def fetch_enverus_data(symbols, startdate, enddate):
    url = (
        f'https://webservice.gvsi.com/api/v3/getdaily?symbols={symbols}'
        f'&fields=close%2Ctradedatetimeutc&output=json&includeheaders=true'
        f'&startdate={startdate}&enddate={enddate}'
    )
    headers = {
        'Authorization': f'{enverus_api_key}'
    }

    try:
        response = requests.request("GET", url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for {symbols}: {e}")
        return None

def process_and_store_trend_data(trend_name, collection_name, data):
    if data and "result" in data and "items" in data["result"]:
        items = data["result"]["items"]
        df = pd.DataFrame(items)

        # Ensure required columns exist
        if 'tradedatetimeutc' in df.columns and 'close' in df.columns:
            # Convert columns to appropriate data types
            df['tradedatetimeutc'] = pd.to_datetime(df['tradedatetimeutc'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')

            # Sort by date and keep the last 14 entries
            df = df.sort_values('tradedatetimeutc').tail(14)

            # Get the collection for the trend
            trend_collection = db[collection_name]

            # Remove old data for this specific trend only
            trend_collection.delete_many({"trend": trend_name})
            logger.info(f"Old data cleared for trend '{trend_name}' in collection '{collection_name}'.")

            # Insert new data
            records = df.to_dict("records")
            trend_collection.insert_many([{
                "date": record['tradedatetimeutc'],
                "Close": record['close'],
                "trend": trend_name
            } for record in records])
            logger.info(f"Data for {trend_name} successfully stored in {collection_name}")
        else:
            logger.warning(f"{trend_name}: Missing required keys in data")
    else:
        logger.warning(f"{trend_name}: 'result' or 'items' key missing in API response")


# Fetch news data from Enverus API
def fetch_news():
    url = (
        'https://webservice.gvsi.com/api/v3/getnews?source=PCR&fields=source%2Cstory%2Cheadline&output=json&includeheaders=true'
    )
    headers = {
        'Authorization': f'{enverus_api_key}'
    }
    payload = """"""

    try:
        # API request
        response = requests.request("GET", url, headers=headers, data=payload)
        response.raise_for_status()

        # Parse JSON response
        response_data = response.json()
        items = response_data["result"]["items"]
        if items is not None:
            return response_data["result"]["items"]
        else:
            logger.warning("No news data found in API response")
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch news: {e}")
        return []

# Process and store news data in MongoDB
def process_and_store_news(news_items):
    if news_items:
        news_collection = db['news']

        # Remove old news
        news_collection.delete_many({})
        logger.info("Old news data cleared")

        # Insert new news data
        for news in news_items:
            news_collection.insert_one({
                "source": news.get("source"),
                "headline": news.get("headline"),
                "story": news.get("story"),
                "timestamp": datetime.now(timezone.utc)  # Add timestamp for when the news was fetched
            })
        logger.info("News data successfully stored in MongoDB")
    else:
        logger.warning("No news items to store")

# Main function to fetch and process data for multiple symbols
def main():
    # Define symbols and trends
    trends = [
        {"trend_name": "Dated Brent Oil Price", "symbols": "%23D.PCAAS00"},
        {"trend_name": "Naphtha Crack Spread", "symbols": "%40Naphtha_Crack"},
        {"trend_name": "Gasoline 95 Crack Spread", "symbols": "%40Gasoline95_Crack"},
        {"trend_name": "Gasoline 97 Crack Spread", "symbols": "%40Gasoline97_Crack"},
        {"trend_name": "Gasoil 10PPM Crack Spread", "symbols": "%40Gasoil10PPM_Crack"},
        {"trend_name": "Gasoil 500PPM Crack Spread", "symbols": "%40Gasoil500PPM_Crack"},
        {"trend_name": "Gasoil 2500PPM Crack Spread", "symbols": "%40Gasoil2500PPM_Crack"},
    ]

    # Fetch and process data for each trend
    for trend in trends:
        trend_name = trend["trend_name"]
        symbols = trend["symbols"]
        collection_name = "prices"

        # Calculate date range (last 14 days)
        end_date = datetime.now().strftime("%m/%d/%Y")
        start_date = (datetime.now() - timedelta(days=14)).strftime("%m/%d/%Y")

        # Fetch data from the API
        logger.info(f"Fetching data for {trend_name} ({symbols}) from {start_date} to {end_date}")
        api_response = fetch_enverus_data(symbols, start_date, end_date)

        # Process the API response and store in MongoDB
        if api_response:
            process_and_store_trend_data(trend_name, collection_name, api_response)
        else:
            logger.warning(f"Failed to fetch data for {trend_name} ({symbols}).")
    
    # Fetch and process news
    logger.info("Fetching news data")
    news_items = fetch_news()
    process_and_store_news(news_items)

# Run the main function
if __name__ == "__main__":
    main()

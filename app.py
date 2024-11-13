import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
from openai import OpenAI
import matplotlib.pyplot as plt
from io import BytesIO
from dotenv import load_dotenv
import os
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# Get API keys and other sensitive data from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
smtp_email = os.getenv("SMTP_EMAIL")
smtp_password = os.getenv("SMTP_PASSWORD")
mongo_uri = os.getenv("MONGO_URI")

# Initialize MongoDB client
client_db = MongoClient(mongo_uri)
db = client_db['crude_oil_analysis']
prices_collection = db['prices']
news_collection = db['news']
client = OpenAI(api_key=openai_api_key)

# 1. Crude Oil Price Grabber Agent (MongoDB)
def crude_oil_price_grabber():
    data = list(prices_collection.find().sort("date", -1).limit(14))
    if data:
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()  # Ensure data is sorted by date ascending
        df = df[["Close"]]  # Get only the 'Close' column
        forecast = forecast_crude_oil_price(df)  # Call ARIMA forecast function
        return df, forecast
    else:
        raise ValueError("Error fetching crude oil data from MongoDB")

# 2. Forecast Crude Oil Prices using ARIMA
def forecast_crude_oil_price(df):
    model = ARIMA(df["Close"], order=(10, 2, 5))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=7)
    return forecast

# 3. Crude Oil Analyst Agent
def crude_oil_analyst(df, forecast):
    trend = "upward" if df["Close"].iloc[-1] > df["Close"].iloc[0] else "downward"
    forecast_trend = "increasing" if forecast.iloc[-1] > df["Close"].iloc[-1] else "decreasing"
    keywords = ["crude oil", "OPEC", "geopolitical", "conflict"] if trend == "upward" else ["crude oil", "supply", "inflation", "demand"]
    
    analysis = {
        "trend": trend,
        "forecast_trend": forecast_trend,
        "keywords": keywords
    }
    return analysis

# 4. News Analyst Agent (MongoDB)
def news_analyst(keywords):
    # Use regex to find articles matching the keywords in title or description without requiring a text index
    articles = list(news_collection.find({
        "$or": [
            {"title": {"$regex": "|".join(keywords), "$options": "i"}},
            {"description": {"$regex": "|".join(keywords), "$options": "i"}}
        ]
    }).sort("publishedAt", -1).limit(7))
    if articles:
        news_summary = "\n\n".join([f"{article['title']}: {article['description']}" for article in articles])
        return news_summary
    else:
        return "No news data available."

# 5. Market Analyst Agent
def market_analyst(df, forecast, news_summary, crude_oil_analysis):
    trend = crude_oil_analysis["trend"]
    forecast_trend = crude_oil_analysis["forecast_trend"]
    latest_price = df["Close"].iloc[-1]
    forecast_price = forecast.iloc[-1]

    prompt = (
        f"The current crude oil price trend over the past two weeks shows a {trend} movement. "
        f"The 7-day forecast indicates prices are likely to be {forecast_trend}. "
        f"The latest price is ${latest_price:.2f}, with a forecasted price of ${forecast_price:.2f}.\n\n"
        f"Relevant News:\n{news_summary}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing crude oil market analysis. Your analysis should be concise and within 4-5 paragraphs"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )

    return response.choices[0].message.content

# Create trend plot and return as BytesIO image
def create_trend_plot(df, forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Close"], label="Historical Prices", color="blue")
    
    future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 8)]
    plt.plot(future_dates, forecast, label="7-Day Forecast", color="orange", linestyle="--")
    
    plt.title("WTI Crude Oil Close Price and Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    return buffer

# Send email with embedded trend plot and analysis
def send_email(subject, analysis, trend_plot_image, recipient_emails):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # Convert recipient list to a comma-separated string for the 'To' field
    recipients_str = ", ".join(recipient_emails)

    msg = MIMEMultipart("related")
    msg["From"] = smtp_email
    msg["To"] = recipients_str
    msg["Subject"] = subject

    # HTML body with Streamlit app link and embedded image
    html = f"""
    <html>
    <body>
        <h2>MEGAT Crude Oil Price Analysis</h2>
        <p>{analysis}</p>
        <h3>Price Trend</h3>
        <img src="cid:trend_plot">
        <br>
        <p>For more detailed insights, visit our <a href="https://megat-analyst.streamlit.app">Streamlit app</a>.</p>
    </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))

    # Attach the trend plot image with Content-ID
    image = MIMEImage(trend_plot_image.read())
    image.add_header('Content-ID', '<trend_plot>')
    msg.attach(image)

    # Send the email to each recipient
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_email, smtp_password)
        server.sendmail(smtp_email, recipient_emails, msg.as_string())

# Main function to execute all agents
def main():
    df, forecast = crude_oil_price_grabber()
    crude_oil_analysis = crude_oil_analyst(df, forecast)
    news_summary = news_analyst(crude_oil_analysis["keywords"])
    analysis = market_analyst(df, forecast, news_summary, crude_oil_analysis)
    trend_plot_image = create_trend_plot(df, forecast)
    
    # Define multiple recipients
    recipients = ["abuhuzaifah.bidin@petronas.com.my"]
    send_email("MEGAT Daily Crude Oil Insights", analysis, trend_plot_image, recipients)

if __name__ == "__main__":
    main()

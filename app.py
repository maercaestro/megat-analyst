import requests
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import openai
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API keys and other sensitive data from environment variables
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
smtp_email = os.getenv("SMTP_EMAIL")
smtp_password = os.getenv("SMTP_PASSWORD")

# 1. Crude Oil Price Grabber Agent
def crude_oil_price_grabber():
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=CL&apikey={alpha_vantage_api_key}&outputsize=compact"
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" in data:
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.rename(columns={"4. close": "Close"})
        df = df[["Close"]].tail(14)  # Get the last 14 days for analysis
        forecast = forecast_crude_oil_price(df)  # Call ARIMA forecast function
        return df, forecast
    else:
        raise ValueError("Error fetching crude oil data")

def forecast_crude_oil_price(df):
    model = ARIMA(df["Close"], order=(10, 2, 5))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=7)
    return forecast

# 2. Crude Oil Analyst Agent
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

# 3. News Analyst Agent
def news_analyst(keywords):
    query = " OR ".join(keywords)
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize=7&sortBy=publishedAt&language=en&apiKey={news_api_key}"
    response = requests.get(url)
    news_data = response.json()

    if news_data.get("status") == "ok":
        articles = news_data["articles"]
        news_summary = "\n\n".join([f"{article['title']}: {article['description']}" for article in articles[:7]])
        return news_summary
    else:
        return "No news data available."

# 4. Market Analyst Agent
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

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI agent providing a concise summary of crude oil price trends, forecasts, and relevant news context in three paragraphs. Please explain why the crude oil price dips/increase based on the news available"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )

    return response['choices'][0]['message']['content']

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
def send_email(subject, analysis, trend_plot_image, recipient_email):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    msg = MIMEMultipart("related")
    msg["From"] = smtp_email
    msg["To"] = recipient_email
    msg["Subject"] = subject

    # HTML body with image reference by Content-ID
    html = f"""
    <html>
    <body>
        <h2>Crude Oil Price Analysis</h2>
        <p>{analysis}</p>
        <h3>Price Trend</h3>
        <img src="cid:trend_plot">
    </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))

    # Attach the trend plot image with Content-ID
    image = MIMEImage(trend_plot_image.read())
    image.add_header('Content-ID', '<trend_plot>')
    msg.attach(image)

    # Send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_email, smtp_password)
        server.sendmail(smtp_email, recipient_email, msg.as_string())

# Main function to execute all agents
def main():
    df, forecast = crude_oil_price_grabber()
    crude_oil_analysis = crude_oil_analyst(df, forecast)
    news_summary = news_analyst(crude_oil_analysis["keywords"])
    analysis = market_analyst(df, forecast, news_summary, crude_oil_analysis)
    trend_plot_image = create_trend_plot(df, forecast)
    
    # Send the email with inline image and analysis
    send_email("Daily Crude Oil Insights", analysis, trend_plot_image, "abuhuzaifah.bidin@petronas.com.my")

if __name__ == "__main__":
    main()

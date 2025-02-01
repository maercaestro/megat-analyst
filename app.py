import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import matplotlib.pyplot as plt
from io import BytesIO
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from openai import OpenAI  # Updated import based on your amendments

# Load environment variables from .env file
load_dotenv()

# Get API keys and other sensitive data from environment variables
smtp_email = os.getenv("SMTP_EMAIL")
smtp_password = os.getenv("SMTP_PASSWORD")
mongo_uri = os.getenv("MONGO_URI")

# Initialize the OpenAI client using your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize MongoDB client
client_db = MongoClient(mongo_uri)
db = client_db['crude_oil_analysis']
news_collection = db['news']

# -------------------------------
# 1. DATA RETRIEVAL & FORECASTING
# -------------------------------

def get_trend_data(collection_name):
    """
    Fetch the last 14 days of data from the specified collection,
    sort by date, and compute a 7-day ARIMA forecast using the same parameters for all trends.
    """
    data = list(db[collection_name].find().sort("date", -1).limit(14))
    if data:
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()  # Ensure ascending date order
        df = df[["Close"]]
        forecast = forecast_trend_price(df)
        return df, forecast
    else:
        raise ValueError(f"Error fetching data from collection: {collection_name}")

def forecast_trend_price(df):
    """Forecast the next 7 days using ARIMA with order (10, 2, 5)."""
    model = ARIMA(df["Close"], order=(10, 2, 5))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=7)
    return forecast

# -------------------------------
# 2. NEWS RETRIEVAL
# -------------------------------

def get_news_summary():
    """Fetch the latest 7 news items and compile a brief summary."""
    news_items = list(news_collection.find().sort("timestamp", -1).limit(7))
    if news_items:
        summary = "\n\n".join(
            [f"{item.get('headline', 'No Headline')}: {item.get('story', '')}" for item in news_items]
        )
        return summary
    else:
        return "No news data available."

# -------------------------------
# 3. GPT-4 ANALYSIS
# -------------------------------

def market_analyst(trends_data, news_summary):
    """
    Build a prompt that summarizes:
      1. Crude oil prices (Dated Brent Oil Price),
      2. Direction for Reforming Margin (using Naphtha and Gasoline crack spreads), and
      3. Direction for Middle Distillates (using all Gasoil crack spreads).
    The prompt is then sent to GPT-4 for a concise analysis.
    """
    # 1. Crude Oil Prices (Dated Brent Oil Price)
    brent = trends_data.get("Dated Brent Oil Price")
    if not brent:
        raise ValueError("Dated Brent Oil Price data missing.")
    brent_latest = brent["df"]["Close"].iloc[-1]
    brent_forecast = brent["forecast"].iloc[-1]

    # 2. Reforming Margin: using Naphtha and Gasoline crack spreads
    naphtha = trends_data.get("Naphtha Crack Spread")
    gasoline95 = trends_data.get("Gasoline 95 Crack Spread")
    gasoline97 = trends_data.get("Gasoline 97 Crack Spread")
    if not (naphtha and gasoline95 and gasoline97):
        raise ValueError("One or more required trends for Reforming Margin analysis missing.")
    naphtha_latest = naphtha["df"]["Close"].iloc[-1]
    naphtha_forecast = naphtha["forecast"].iloc[-1]
    gasoline95_latest = gasoline95["df"]["Close"].iloc[-1]
    gasoline95_forecast = gasoline95["forecast"].iloc[-1]
    gasoline97_latest = gasoline97["df"]["Close"].iloc[-1]
    gasoline97_forecast = gasoline97["forecast"].iloc[-1]

    # 3. Middle Distillates: using Gasoil crack spreads (10PPM, 500PPM, 2500PPM)
    gasoil10 = trends_data.get("Gasoil 10PPM Crack Spread")
    gasoil500 = trends_data.get("Gasoil 500PPM Crack Spread")
    gasoil2500 = trends_data.get("Gasoil 2500PPM Crack Spread")
    if not (gasoil10 and gasoil500 and gasoil2500):
        raise ValueError("One or more required trends for Middle Distillates analysis missing.")
    gasoil10_latest = gasoil10["df"]["Close"].iloc[-1]
    gasoil10_forecast = gasoil10["forecast"].iloc[-1]
    gasoil500_latest = gasoil500["df"]["Close"].iloc[-1]
    gasoil500_forecast = gasoil500["forecast"].iloc[-1]
    gasoil2500_latest = gasoil2500["df"]["Close"].iloc[-1]
    gasoil2500_forecast = gasoil2500["forecast"].iloc[-1]

    prompt = (
        f"Analyze the following trends in the crude oil market and provide a concise analysis (4-5 paragraphs) following the structure below:\n\n"
        f"1. Crude Oil Prices:\n"
        f"   - Dated Brent Oil Price: Latest price is ${brent_latest:.2f}, forecasted to ${brent_forecast:.2f}.\n"
        f"   - Based on the above and the following news summary, make a general statement about crude oil prices.\n\n"
        f"2. Direction for Reforming Margin:\n"
        f"   - Naphtha Crack Spread: Latest ${naphtha_latest:.2f}, forecasted ${naphtha_forecast:.2f}.\n"
        f"   - Gasoline 95 Crack Spread: Latest ${gasoline95_latest:.2f}, forecasted ${gasoline95_forecast:.2f}.\n"
        f"   - Gasoline 97 Crack Spread: Latest ${gasoline97_latest:.2f}, forecasted ${gasoline97_forecast:.2f}.\n"
        f"   - Based on the above, what is the likely direction for reforming margin?\n\n"
        f"3. Direction for Middle Distillates:\n"
        f"   - Gasoil 10PPM Crack Spread: Latest ${gasoil10_latest:.2f}, forecasted ${gasoil10_forecast:.2f}.\n"
        f"   - Gasoil 500PPM Crack Spread: Latest ${gasoil500_latest:.2f}, forecasted ${gasoil500_forecast:.2f}.\n"
        f"   - Gasoil 2500PPM Crack Spread: Latest ${gasoil2500_latest:.2f}, forecasted ${gasoil2500_forecast:.2f}.\n"
        f"   - Based on the above, what is the likely direction for middle distillates?\n\n"
        f"NEWS SUMMARY:\n{news_summary}\n\n"
        f"Please provide your analysis in a structured, concise format as specified."
    )

    # Use the updated GPT-4 API call
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing market analysis."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=350,
        temperature=0.7
    )

    return response.choices[0].message.content

# -------------------------------
# 4. CHART PLOTTING
# -------------------------------

def create_crude_oil_chart(trends_data):
    """Plot the crude oil trend (Dated Brent Oil Price) with historical data and forecast."""
    brent = trends_data.get("Dated Brent Oil Price")
    if not brent:
        raise ValueError("Dated Brent Oil Price data missing.")
    
    df = brent["df"]
    forecast = brent["forecast"]
    
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Close"], label="Historical Prices", color="blue")
    future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 8)]
    plt.plot(future_dates, forecast, label="7-Day Forecast", color="orange", linestyle="--")
    
    plt.title("Crude Oil Trend (Dated Brent Oil Price)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    return buffer

def create_cracks_chart(trends_data):
    """
    Plot the crack spreads with historical data and forecast.
    This includes Naphtha, Gasoline 95, Gasoline 97, Gasoil 10PPM, Gasoil 500PPM, and Gasoil 2500PPM Crack Spreads.
    """
    crack_keys = [
        "Naphtha Crack Spread", 
        "Gasoline 95 Crack Spread", 
        "Gasoline 97 Crack Spread", 
        "Gasoil 10PPM Crack Spread", 
        "Gasoil 500PPM Crack Spread", 
        "Gasoil 2500PPM Crack Spread"
    ]
    
    plt.figure(figsize=(12, 8))
    for key in crack_keys:
        trend = trends_data.get(key)
        if trend:
            df = trend["df"]
            forecast = trend["forecast"]
            plt.plot(df.index, df["Close"], label=f"{key} Historical")
            future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 8)]
            plt.plot(future_dates, forecast, linestyle="--", label=f"{key} Forecast")
    
    plt.title("Crack Spreads Trend")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend(fontsize=8)
    plt.grid(True)
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    return buffer

# -------------------------------
# 5. EMAIL SENDING
# -------------------------------

def send_email(subject, analysis, crude_chart_image, cracks_chart_image, recipient_emails):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    recipients_str = ", ".join(recipient_emails)

    msg = MIMEMultipart("related")
    msg["From"] = smtp_email
    msg["To"] = recipients_str
    msg["Subject"] = subject

    # Replace newline characters with HTML breaks for improved formatting
    formatted_analysis = analysis.replace("\n", "<br>")

    # Construct HTML with clear headers and both charts
    html = f"""
    <html>
    <body>
        <h2>MEGAT Daily Crude Oil Insights</h2>
        <h3>Market Analysis</h3>
        <div style="font-family: Arial, sans-serif; line-height: 1.5;">
            {formatted_analysis}
        </div>
        <hr>
        <h3>Crude Oil Price Trend (Dated Brent Oil Price)</h3>
        <img src="cid:crude_chart" style="max-width: 600px;"><br>
        <hr>
        <h3>Crack Spreads Trend</h3>
        <img src="cid:cracks_chart" style="max-width: 600px;"><br>
        <p>For more detailed insights, visit our <a href="https://megat-analyst.streamlit.app">Streamlit app</a>.</p>
    </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))

    # Attach the crude oil chart image
    crude_img = MIMEImage(crude_chart_image.read())
    crude_img.add_header('Content-ID', '<crude_chart>')
    msg.attach(crude_img)

    # Attach the cracks chart image
    cracks_img = MIMEImage(cracks_chart_image.read())
    cracks_img.add_header('Content-ID', '<cracks_chart>')
    msg.attach(cracks_img)

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_email, smtp_password)
        server.sendmail(smtp_email, recipient_emails, msg.as_string())

# -------------------------------
# 6. MAIN FUNCTION
# -------------------------------

def main():
    # Define the 7 trends with their corresponding MongoDB collections.
    trends = [
        {"name": "Dated Brent Oil Price", "collection": "prices_dated_brent_oil_price"},
        {"name": "Naphtha Crack Spread", "collection": "prices_naphtha_crack_spread"},
        {"name": "Gasoline 95 Crack Spread", "collection": "prices_gasoline_95_crack_spread"},
        {"name": "Gasoline 97 Crack Spread", "collection": "prices_gasoline_97_crack_spread"},
        {"name": "Gasoil 10PPM Crack Spread", "collection": "prices_gasoil_10ppm_crack_spread"},
        {"name": "Gasoil 500PPM Crack Spread", "collection": "prices_gasoil_500ppm_crack_spread"},
        {"name": "Gasoil 2500PPM Crack Spread", "collection": "prices_gasoil_2500ppm_crack_spread"}
    ]

    trends_data = {}
    for trend in trends:
        try:
            df, forecast = get_trend_data(trend["collection"])
            trends_data[trend["name"]] = {"df": df, "forecast": forecast}
        except ValueError as e:
            print(e)
            return

    news_summary = get_news_summary()
    analysis = market_analyst(trends_data, news_summary)
    
    # Create the two separate charts
    crude_chart_image = create_crude_oil_chart(trends_data)
    cracks_chart_image = create_cracks_chart(trends_data)

    recipients = ["abuhuzaifah.bidin@petronas.com.my"]
    send_email("MEGAT Daily Crude Oil Insights", analysis, crude_chart_image, cracks_chart_image, recipients)

if __name__ == "__main__":
    main()

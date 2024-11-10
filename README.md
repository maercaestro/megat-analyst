

# MEGAT Crude Oil Analysis & Insights using Crew AI and AI Agents

A project that automates daily crude oil price analysis and insights, using a Streamlit app and an email notification system. The app fetches crude oil prices, forecasts trends, and provides analysis with insights from related news. Daily email reports are sent with embedded visualizations and a link to the Streamlit app for further exploration.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Setup](#setup)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [File Structure](#file-structure)
7. [Dependencies](#dependencies)
8. [Acknowledgements](#acknowledgements)

---

## Overview
This project provides automated analysis and insights on crude oil prices:
- **Streamlit App**: A web application to view real-time data, forecasts, and news analysis.
- **Email Automation**: Sends daily emails with trend analysis, forecasts, and relevant news.

---

## Features

- **Crude Oil Price Fetching**: Retrieves historical crude oil price data.
- **Trend Forecasting**: Uses ARIMA to forecast future prices.
- **News Analysis**: Gathers relevant news to provide context for price trends.
- **Streamlit App**: Interactive app with trend visualization and analysis summaries.
- **Email Reports**: Sends daily reports with embedded visualizations and a link to the Streamlit app.

---

## Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/crude-oil-analysis.git
   cd crude-oil-analysis
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**:

   Create a `.env` file in the root directory to store API keys and email credentials.

   ```plaintext
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   NEWS_API_KEY=your_news_api_key
   OPENAI_API_KEY=your_openai_key
   SMTP_EMAIL=your_email@example.com
   SMTP_PASSWORD=your_email_password
   ```

4. **Run the Streamlit App**:

   ```bash
   streamlit run app.py
   ```

5. **Run the Email Automation Script**:

   To manually trigger the email automation:

   ```bash
   python app.py
   ```

---

## Configuration

### Streamlit App URL (for Email Links)
The Streamlit app URL is embedded directly in the email template. Modify it in `app.py` within the `send_email` function if needed:

```python
<p>For more detailed insights, visit our <a href="https://your-streamlit-app-url.com">Streamlit app</a>.</p>
```

### Add Recipients
In the `main()` function, modify the `recipients` list to include multiple recipients for the daily email:

```python
recipients = ["recipient1@example.com", "recipient2@example.com"]
```

---

## Usage

### Streamlit App
The Streamlit app allows users to view crude oil price trends and forecasts, along with relevant news. The interactive dashboard provides insights through charts and summaries.

1. Start the app with `streamlit run app.py`.
2. Access the app at `http://localhost:8501` or your deployment URL if hosted.

### Email Automation
The email automation script runs a full analysis and sends a daily email summary, including:
- Price trends and forecasted movements
- Analysis based on recent news
- Link to the Streamlit app for detailed insights

You can set up a cron job to automate this daily. Here’s an example for running the script every day at 8 AM:

```bash
0 8 * * * /usr/bin/python3 /path/to/your/app.py
```

---

## File Structure

```plaintext
crude-oil-analysis/
├── app.py                     # Main application with Streamlit and email automation
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables for API keys and credentials
├── README.md                  # Project documentation
└── config.toml                # (Optional) Configurations in TOML format
```

---

## Dependencies

- `streamlit`: For running the web app
- `pandas`: Data manipulation
- `requests`: API requests for data fetching
- `openai`: For AI-powered market analysis
- `statsmodels`: ARIMA forecasting model
- `matplotlib`: For data visualization in emails
- `dotenv`: Loads environment variables from a `.env` file
- `smtplib` and `email`: For email automation
- 'crewai' : For managing multiple AI agents

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Acknowledgements

- **Alpha Vantage API** for crude oil price data
- **News API** for fetching relevant news articles
- **OpenAI API** for AI-driven analysis and summarization

!

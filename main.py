__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
from pymongo import MongoClient
from crew_setup import setup_crew
import plotly.graph_objects as go

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()

mongo_uri = os.getenv("MONGO_URI")  # MongoDB URI
client_db = MongoClient(mongo_uri)
db = client_db['crude_oil_analysis']
queries_collection = db['queries']  # Collection for storing queries and responses

# Set up Streamlit page configuration
st.set_page_config(layout="wide")
st.title("Crude Oil Analysis by MEGAT AI Agents")

# Initialize Crew and get the analysis summary along with data, forecast, and news summary
crew, analysis_summary, data, forecast, news_summary = setup_crew()

# Check if crew and analysis summary are available
if crew is None or analysis_summary is None:
    st.error("Failed to initialize the AI analysis. Please check the API keys and network connectivity.")
else:
    # Set up columns for displaying analysis and chat interface
    col1, col2 = st.columns([1, 2])

    # Column 2: Data display and analysis summary
    with col2:
        st.subheader("Market Analysis")

        if data is not None and forecast is not None:
            # Prepare data for the chart
            historical_data = data["Close"].to_list()
            future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=len(forecast))
            forecast_data = forecast.to_list()

            # Ensure that both historical data and forecast data have matching lengths
            if len(historical_data) + len(future_dates) == len(historical_data) + len(forecast_data):
                # Combine historical and forecast data for chart
                chart_data = pd.DataFrame({
                    "Date": data.index.tolist() + future_dates.tolist(),
                    "Close": historical_data + forecast_data,
                    "Type": ["Historical"] * len(historical_data) + ["Forecast"] * len(forecast_data)
                }).set_index("Date")

                # Display the chart and analysis summary using Plotly for interactivity
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=chart_data[chart_data['Type'] == 'Historical'].index, 
                                         y=chart_data[chart_data['Type'] == 'Historical']['Close'], 
                                         mode='lines',
                                         name='Historical', 
                                         line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=chart_data[chart_data['Type'] == 'Forecast'].index, 
                                         y=chart_data[chart_data['Type'] == 'Forecast']['Close'], 
                                         mode='lines',
                                         name='Forecast',
                                         line=dict(color='red')))
                fig.update_layout(title="Historical and Forecasted Crude Oil Prices",
                                  xaxis_title="Date",
                                  yaxis_title="Close Price",
                                  hovermode="x unified",
                                  width = 1000,
                                  height = 600)
                st.plotly_chart(fig)
                st.write(analysis_summary)
            else:
                st.warning("Data lengths for historical and forecast values do not match.")
        else:
            st.warning("Unable to display chart due to missing data.")

 # Column 1: Chat interface using ChatAgent from Crew Setup
    with col1:
        st.subheader("Ask the AI about Crude Oil Trends")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Your question:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Run chat task using ChatAgent from Crew Setup
            chat_agent = crew.agents[1]  # Assuming ChatAgent is the second agent in the crew
            full_response = chat_agent.run(prompt, analysis_summary)

            with st.chat_message("assistant"):
                st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})


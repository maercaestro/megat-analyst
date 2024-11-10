# main.py

import streamlit as st
import pandas as pd
from datetime import timedelta
from crew_setup import setup_crew

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
            future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=7)
            forecast_data = forecast.to_list()

            chart_data = pd.DataFrame({
                "Date": data.index.tolist() + future_dates.tolist(),
                "Close": historical_data + forecast_data
            }).set_index("Date")

            # Display the chart and analysis summary
            st.line_chart(chart_data)
            st.write(analysis_summary)
        else:
            st.warning("Unable to display chart due to missing data.")

    # Column 1: Chat interface
    with col1:
        st.subheader("Ask the AI about Crude Oil Trends")
        user_input = st.text_input("Your question:")

        if st.button("Send"):
            if user_input:
                # Run chat task using the provided analysis summary
                chat_agent = crew.agents[1]  # Assuming ChatAgent is the second agent in the crew
                chat_response = chat_agent.run(user_input, analysis_summary)
                st.write("AI:", chat_response)
            else:
                st.warning("Please enter a question to ask the AI.")

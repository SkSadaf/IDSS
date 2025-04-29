

## Setup Guide

### Step 1: Create a Virtual Environment
First, create a virtual environment for the project:
python -m venv fin_env

### Step 2: Activate the Virtual Environment
Activate the virtual environment:

#### On Windows:
fin_env\Scripts\activate
#### On macOS/Linux:
source fin_env/bin/activate

### Step 3: Install Dependencies
#### Install the required Python packages:

pip install streamlit transformers torch sentence-transformers faiss-cpu plotly pandas numpy requests python-dotenv yfinance matplotlib

### Step 4: Configure API Keys
#### Add your API keys to .env file:

FINNHUB_API_KEY=your_finnhub_api_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_api_key_here
NEWS_API_KEY=your_news_api_key_here

### Step 5: Run the Application
#### Run the Streamlit application:

streamlit run main.py

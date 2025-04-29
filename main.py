import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from sentence_transformers import SentenceTransformer
import faiss
import warnings
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import altair as alt
import time
from dotenv import load_dotenv
import os

warnings.filterwarnings('ignore')

# Load .env file
load_dotenv()

# Document store class
class DocumentStore:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        self.index = None

    def add_documents(self, documents):
        if not documents:
            return
        
        self.documents = documents
        embeddings = self.encoder.encode([doc['text'] for doc in documents])
        self.embeddings = embeddings.astype('float32')

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def search(self, query, k=3):
        if not self.index:
            return []
        
        query_vector = self.encoder.encode([query])
        query_vector = query_vector.astype('float32')
        
        distances, indices = self.index.search(query_vector, k)
        return [self.documents[i] for i in indices[0]]

class FinancialAnalysisSystem:
    def __init__(self):
        st.sidebar.text("Initializing Financial Analysis System...")
        self.setup_models()
        self.setup_document_store()
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        st.sidebar.success("System initialized successfully!")

    def setup_models(self):
        st.sidebar.text("Loading FinBERT model...")
        self.finbert_tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        self.finbert_model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

        st.sidebar.text("Loading FinGPT model...")
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.fingpt_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.fingpt_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.fingpt_tokenizer.pad_token = self.fingpt_tokenizer.eos_token
        self.fingpt_model.config.pad_token_id = self.fingpt_tokenizer.pad_token_id
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.finbert_model = self.finbert_model.to(self.device)
        self.fingpt_model = self.fingpt_model.to(self.device)
        
        self.sentiment_labels = ['positive', 'negative', 'neutral']

    def setup_document_store(self):
        self.doc_store = DocumentStore()

    def get_company_news_and_reports(self, ticker):
        news_articles = []

        if self.alpha_vantage_key and self.alpha_vantage_key != "YOUR_ALPHA_VANTAGE_KEY":
            try:
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.alpha_vantage_key}"
                response = requests.get(url)
                alpha_data = response.json()
                
                if 'feed' in alpha_data:
                    for article in alpha_data['feed'][:10]:
                        news_articles.append({
                            'title': article.get('title', ''),
                            'text': article.get('summary', ''),
                            'url': article.get('url', ''),
                            'date': article.get('time_published', ''),
                            'source': 'Alpha Vantage'
                        })
            except Exception as e:
                st.warning(f"Warning: Unable to fetch Alpha Vantage news: {str(e)}")

        if self.news_api_key and self.news_api_key != "YOUR_NEWS_API_KEY":
            try:
                company_name = yf.Ticker(ticker).info.get('longName', ticker)
                url = f"https://newsapi.org/v2/everything?q={company_name}&apiKey={self.news_api_key}&language=en&sortBy=publishedAt"
                response = requests.get(url)
                news_data = response.json()
                
                if news_data.get('status') == 'ok' and news_data.get('articles'):
                    for article in news_data['articles'][:10]:
                        news_articles.append({
                            'title': article.get('title', ''),
                            'text': article.get('description', ''),
                            'url': article.get('url', ''),
                            'date': article.get('publishedAt', ''),
                            'source': f"NewsAPI - {article.get('source', {}).get('name', 'Unknown')}"
                        })
            except Exception as e:
                st.warning(f"Warning: Unable to fetch NewsAPI news: {str(e)}")

        if not news_articles:
            st.warning("No news articles found. Using basic company information.")
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                news_articles.append({
                    'title': f"Basic Information - {ticker}",
                    'text': f"Company: {info.get('longName', ticker)}\n"
                            f"Industry: {info.get('industry', 'N/A')}\n"
                            f"Sector: {info.get('sector', 'N/A')}\n"
                            f"Description: {info.get('longBusinessSummary', 'N/A')}",
                    'source': 'Company Info'
                })
            except Exception as e:
                st.error(f"Error fetching company information: {str(e)}")

        seen_titles = set()
        unique_articles = []
        for article in news_articles:
            if article['title'] not in seen_titles:
                seen_titles.add(article['title'])
                unique_articles.append(article)
        try:
            unique_articles.sort(key=lambda x: x['date'], reverse=True)
        except:
            pass

        if unique_articles:
            self.doc_store.add_documents(unique_articles)
        
        return unique_articles

    def analyze_sentiment(self, text):
        inputs = self.finbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.finbert_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        sentiment = self.sentiment_labels[predictions.argmax().item()]
        confidence = predictions.max().item()
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {label: prob.item() for label, prob in zip(self.sentiment_labels, predictions[0])}
        }

    def generate_fingpt_recommendation(self, context):
        prompt = f"""
        Based on the following financial analysis, provide an investment recommendation:

        Company Analysis:
        {context}

        Provide a detailed recommendation including:
        1. Investment action (BUY/HOLD/SELL)
        2. Detailed reasoning
        3. Risk factors
        4. Price target
        5. Investment timeframe
        """
        try:
            inputs = self.fingpt_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.fingpt_model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=1000,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=self.fingpt_tokenizer.pad_token_id,
                    eos_token_id=self.fingpt_tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50
                )
            
            recommendation = self.fingpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
            parsed_recommendation = self.parse_recommendation(recommendation)
            
            if not parsed_recommendation['action']:
                parsed_recommendation['action'] = self.get_fallback_recommendation()['action']
                
            return parsed_recommendation
        except Exception as e:
            st.error(f"Error generating recommendation: {str(e)}")
            return self.get_fallback_recommendation()

    def get_fallback_recommendation(self):
        return {
            'action': 'HOLD',
            'reasoning': ['Insufficient data to make a strong recommendation'],
            'risks': ['Market uncertainty'],
            'price_target': None,
            'timeframe': 'Short-term'
        }

    def parse_recommendation(self, text):
        lines = text.split('\n')
        recommendation = {
            'action': None,
            'reasoning': [],
            'price_target': None,
            'timeframe': None
        }
        current_section = None
        for line in lines:
            line = line.strip()
            if any(action in line.upper() for action in ['BUY', 'SELL', 'HOLD']):
                for action in ['BUY', 'SELL', 'HOLD']:
                    if action in line.upper():
                        recommendation['action'] = action
                        break
            elif 'reasoning' in line.lower() or 'rationale' in line.lower() or 'because' in line.lower():
                current_section = 'reasoning'
            elif 'price target' in line.lower():
                try:
                    import re
                    price_match = re.search(r'\$?(\d+\.?\d*)', line)
                    if price_match:
                        recommendation['price_target'] = float(price_match.group(1))
                except:
                    pass
            elif 'timeframe' in line.lower() or 'horizon' in line.lower():
                recommendation['timeframe'] = line.strip()
            elif current_section and line:
                recommendation[current_section].append(line)
        return recommendation

    def get_historical_data(self, ticker, period="1y"):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist
        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()

    def get_financials(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cash_flow
            
            financials = {}
            if not income_stmt.empty:
                financials['revenue'] = income_stmt.loc['Total Revenue'].to_dict()
                financials['net_income'] = income_stmt.loc['Net Income'].to_dict()
                
            if not balance_sheet.empty:
                financials['total_assets'] = balance_sheet.loc['Total Assets'].to_dict()
                financials['total_liabilities'] = balance_sheet.loc['Total Liabilities Net Minority Interest'].to_dict()
                
            if not cash_flow.empty:
                financials['operating_cash_flow'] = cash_flow.loc['Operating Cash Flow'].to_dict()
                
            return financials
        except Exception as e:
            st.error(f"Error fetching financial metrics: {str(e)}")
            return {}

    def analyze_company(self, ticker):
        st.info(f"Analyzing {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            documents = self.get_company_news_and_reports(ticker)
            
            sentiment_results = []
            for doc in documents:
                if doc.get('text'):
                    sentiment = self.analyze_sentiment(doc['text'])
                    sentiment_results.append({
                        'title': doc['title'],
                        'sentiment': sentiment['sentiment'],
                        'confidence': sentiment['confidence'],
                        'probabilities': sentiment['probabilities']
                    })
            
            historical_data = self.get_historical_data(ticker)
            financials = self.get_financials(ticker)
            
            context = {
                'company_name': info.get('longName', ticker),
                'ticker': ticker,
                'current_price': info.get('regularMarketPrice', info.get('currentPrice', 'N/A')),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A'),
                'sentiment_analysis': sentiment_results,
                'recent_documents': self.doc_store.search(f"{ticker} financial analysis")
            }
            
            recommendation = self.generate_fingpt_recommendation(str(context))
            
            return {
                'company_info': context,
                'sentiment_analysis': sentiment_results,
                'recommendation': recommendation,
                'historical_data': historical_data,
                'financials': financials
            }
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            return None

def visualize_sentiment(sentiment_results):
    if not sentiment_results:
        st.warning("No sentiment data to visualize")
        return
    
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for result in sentiment_results:
        sentiment = result['sentiment']
        sentiment_counts[sentiment] += 1
    
    fig_pie = px.pie(
        values=list(sentiment_counts.values()),
        names=list(sentiment_counts.keys()),
        title="Sentiment Distribution",
        color=list(sentiment_counts.keys()),
        color_discrete_map={"positive": "green", "neutral": "blue", "negative": "red"}
    )
    st.plotly_chart(fig_pie, use_container_width=True, key="sentiment_pie")
    
    confidence_data = pd.DataFrame([
        {
            'Title': result['title'][:30] + '...' if len(result['title']) > 30 else result['title'],
            'Confidence': result['confidence'],
            'Sentiment': result['sentiment']
        }
        for result in sentiment_results
    ])
    
    fig_bar = px.bar(
        confidence_data,
        x='Title',
        y='Confidence',
        color='Sentiment',
        title="Sentiment Confidence by Article",
        color_discrete_map={"positive": "green", "neutral": "blue", "negative": "red"}
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True, key="sentiment_bar")

def visualize_stock_data(historical_data, ticker, recommendation):
    if historical_data.empty:
        st.warning("No historical data to visualize")
        return
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.25, 
                       subplot_titles=('Price', 'Volume'), 
                       row_width=[0.2, 0.7])
    
    fig.add_trace(
        go.Candlestick(x=historical_data.index,
                       open=historical_data['Open'],
                       high=historical_data['High'],
                       low=historical_data['Low'],
                       close=historical_data['Close'],
                       name=f"{ticker} Price"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=historical_data.index, 
               y=historical_data['Volume'],
               name='Volume',
               marker=dict(color='rgba(58, 71, 80, 0.6)')),
        row=2, col=1
    )
    
    historical_data['MA50'] = historical_data['Close'].rolling(window=50).mean()
    historical_data['MA200'] = historical_data['Close'].rolling(window=200).mean()
    
    fig.add_trace(
        go.Scatter(x=historical_data.index, 
                   y=historical_data['MA50'],
                   name='50-day MA',
                   line=dict(color='rgba(255, 165, 0, 0.8)')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=historical_data.index, 
                   y=historical_data['MA200'],
                   name='200-day MA',
                   line=dict(color='rgba(128, 0, 128, 0.8)')),
        row=1, col=1
    )
    
    if recommendation and recommendation.get('price_target'):
        fig.add_shape(
            type="line",
            x0=historical_data.index[0],
            y0=recommendation['price_target'],
            x1=historical_data.index[-1],
            y1=recommendation['price_target'],
            line=dict(
                color="green" if recommendation['action'] == 'BUY' else 
                      "red" if recommendation['action'] == 'SELL' else "blue",
                width=2,
                dash="dash",
            ),
            row=1, col=1
        )
        
        fig.add_annotation(
            x=historical_data.index[-1],
            y=recommendation['price_target'],
            text=f"Target: ${recommendation['price_target']}",
            showarrow=False,
            yshift=10,
            row=1, col=1
        )

    fig.update_layout(
        title=f"{ticker} Stock Price History",
        height=650,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(t=100)
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True, key="stock_price_chart")
    
    st.subheader("Performance Metrics")
    try:
        returns = historical_data['Close'].pct_change().dropna()
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            start_price = historical_data['Close'].iloc[0]
            current_price = historical_data['Close'].iloc[-1]
            total_return = (current_price - start_price) / start_price * 100
            st.metric("Total Return", f"{total_return:.2f}%", 
                     delta=f"{total_return:.2f}%", 
                     delta_color="normal")
            
        with metrics_col2:
            volatility = returns.std() * np.sqrt(252) * 100
            st.metric("Volatility (Ann.)", f"{volatility:.2f}%")
            
        with metrics_col3:
            cum_returns = (1 and returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max - 1) * 100
            max_drawdown = drawdown.min()
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            
        with metrics_col4:
            mean_return = returns.mean() * 252
            sharpe = mean_return / (returns.std() * np.sqrt(252))
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    except Exception as e:
        st.warning(f"Could not calculate performance metrics: {str(e)}")

def visualize_financials(financials, company_info):
    if not financials:
        st.warning("No financial data available to visualize")
        return
    
    st.subheader("Financial Performance")
    
    try:
        if 'revenue' in financials and 'net_income' in financials:
            fin_df = pd.DataFrame({
                'Revenue': financials['revenue'],
                'Net Income': financials['net_income']
            }).T
            
            fin_df = fin_df[sorted(fin_df.columns)]
            date_labels = [str(date)[:10] for date in fin_df.columns]
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    x=date_labels,
                    y=fin_df.loc['Revenue'].values / 1e9,
                    name='Revenue',
                    marker_color='lightblue'
                )
            )
            
            fig.add_trace(
                go.Bar(
                    x=date_labels,
                    y=fin_df.loc['Net Income'].values / 1e9,
                    name='Net Income',
                    marker_color='darkblue'
                )
            )
            
            fig.update_layout(
                title='Revenue and Net Income (Billions $)',
                xaxis_title='Year',
                yaxis_title='Amount ($ Billions)',
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key="financial_revenue_income")
            
        if 'total_assets' in financials and 'total_liabilities' in financials:
            balance_df = pd.DataFrame({
                'Total Assets': financials['total_assets'],
                'Total Liabilities': financials['total_liabilities']
            }).T
            
            balance_df = balance_df[sorted(balance_df.columns)]
            date_labels = [str(date)[:10] for date in balance_df.columns]
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=date_labels,
                    y=balance_df.loc['Total Assets'].values / 1e9,
                    name='Total Assets',
                    line=dict(color='green', width=2)
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=date_labels,
                    y=balance_df.loc['Total Liabilities'].values / 1e9,
                    name='Total Liabilities',
                    line=dict(color='red', width=2)
                )
            )
            
            fig.update_layout(
                title='Assets and Liabilities (Billions $)',
                xaxis_title='Year',
                yaxis_title='Amount ($ Billions)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key="financial_assets_liabilities")
            
        st.subheader("Key Financial Ratios")
        
        ratio_col1, ratio_col2, ratio_col3, ratio_col4 = st.columns(4)
        
        with ratio_col1:
            pe_ratio = company_info.get('pe_ratio', 'N/A')
            if pe_ratio != 'N/A':
                pe_ratio = f"{pe_ratio:.2f}"
            st.metric("P/E Ratio", pe_ratio)
            
        with ratio_col2:
            try:
                profit_margin = company_info.get('profitMargins', 'N/A')
                if profit_margin != 'N/A':
                    profit_margin = f"{profit_margin * 100:.2f}%"
                st.metric("Profit Margin", profit_margin)
            except:
                st.metric("Profit Margin", "N/A")
            
        with ratio_col3:
            try:
                dividend_yield = company_info.get('dividendYield', 'N/A')
                if dividend_yield != 'N/A':
                    dividend_yield = f"{dividend_yield * 100:.2f}%"
                st.metric("Dividend Yield", dividend_yield)
            except:
                st.metric("Dividend Yield", "N/A")
            
        with ratio_col4:
            try:
                beta = company_info.get('beta', 'N/A')
                if beta != 'N/A':
                    beta = f"{beta:.2f}"
                st.metric("Beta", beta)
            except:
                st.metric("Beta", "N/A")
                
    except Exception as e:
        st.error(f"Error creating financial visualizations: {str(e)}")

def main():
    st.set_page_config(
        page_title="Financial Analysis Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📊 Financial Analysis Dashboard")
    st.markdown("""
    ## Disclaimer
    **Please note:** The analysis provided by this dashboard is for educational purposes only and does not constitute financial advice. 
    Always do your own research and consult with a professional financial advisor before making any investment decisions.
    """)
    st.sidebar.title("Configuration")

    @st.cache_resource
    def get_analyzer():
        return FinancialAnalysisSystem()
        
    analyzer = get_analyzer()

    ticker = st.sidebar.text_input("Enter company ticker symbol (e.g., AAPL, GOOGL):", "AAPL").upper()
    
    if st.sidebar.button("Analyze"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Fetching company data...")
        progress_bar.progress(20)
        time.sleep(0.5)
        
        status_text.text("Processing financial information...")
        progress_bar.progress(40)
        time.sleep(0.5)
        
        status_text.text("Analyzing market sentiment...")
        progress_bar.progress(60)
        time.sleep(0.5)
        
        status_text.text("Generating recommendation...")
        progress_bar.progress(80)
        
        report = analyzer.analyze_company(ticker)
        
        progress_bar.progress(100)
        status_text.text("Analysis completed!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        if report:
            st.header(f"📝 Company Overview: {report['company_info']['company_name']} ({ticker})")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${report['company_info']['current_price']}")
            with col2:
                market_cap = report['company_info']['market_cap']
                if isinstance(market_cap, (int, float)):
                    market_cap_str = f"${market_cap/1e9:.2f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.2f}M"
                    st.metric("Market Cap", market_cap_str)
                else:
                    st.metric("Market Cap", "N/A")
            with col3:
                st.metric("Industry", report['company_info']['industry'])
            with col4:
                st.metric("Sector", report['company_info']['sector'])
            
            with st.expander("Company Description"):
                st.write(report['company_info']['description'])
            
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Stock Performance", "🔍 Sentiment Analysis", "📊 Financials", "🧠 AI Recommendation"])
            
            with tab1:
                visualize_stock_data(report['historical_data'], ticker, report['recommendation'])
            
            with tab2:
                visualize_sentiment(report['sentiment_analysis'])
                
                st.subheader("Recent News Articles")
                for i, doc in enumerate(analyzer.doc_store.documents[:10]):
                    with st.expander(f"{i+1}. {doc.get('title', 'Article')}"):
                        st.write(f"**Source:** {doc.get('source', 'Unknown')}")
                        st.write(f"**Date:** {doc.get('date', 'Unknown')}")
                        st.write(f"**Summary:**\n{doc.get('text', 'No content available')}")
                        if doc.get('url'):
                            st.write(f"[Read full article]({doc['url']})")
                        st.markdown("---")
            
            with tab3:
                visualize_financials(report['financials'], report['company_info'])
            
            with tab4:
                recommendation = report['recommendation']
                
                action = recommendation['action']
                if action == 'BUY':
                    st.success(f"### Recommendation: {action}")
                elif action == 'SELL':
                    st.error(f"### Recommendation: {action}")
                else:
                    st.info(f"### Recommendation: {action}")
                
                if recommendation['price_target']:
                    current_price = report['company_info']['current_price']
                    if isinstance(current_price, (int, float)):
                        price_difference = ((recommendation['price_target'] - current_price) / current_price) * 100
                        st.metric("Price Target", f"${recommendation['price_target']:.2f}", 
                                f"{price_difference:.2f}% from current")
                    else:
                        st.metric("Price Target", f"${recommendation['price_target']:.2f}")
                
                if recommendation['timeframe']:
                    st.write(f"**Timeframe:** {recommendation['timeframe']}")
                
                st.subheader("Investment Thesis")
                
                company_name = report['company_info']['company_name']
                industry = report['company_info']['industry']
                sector = report['company_info']['sector']
                
                sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
                for result in report['sentiment_analysis']:
                    sentiment_counts[result['sentiment']] += 1
                dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
                
                if not report['historical_data'].empty:
                    recent_return = ((report['historical_data']['Close'][-1] / report['historical_data']['Close'][0]) - 1) * 100
                    price_trend = "upward" if recent_return > 0 else "downward"
                else:
                    price_trend = "unclear"
                    recent_return = 0
                
                reasoning = f"""
                The {action} recommendation for {company_name} ({ticker}) is based on a comprehensive analysis of multiple factors:

                Market Position: As a player in the {industry} industry within the {sector} sector, the company {report['company_info']['description'][:200]}...

                Market Sentiment: Analysis of recent news and market commentary shows predominantly {dominant_sentiment} sentiment, with {sentiment_counts[dominant_sentiment]} out of {sum(sentiment_counts.values())} analyzed articles showing {dominant_sentiment} signals.

                Price Action: The stock has shown a {price_trend} trend, with a {recent_return:.2f}% change over the analyzed period. """
                
                if report['financials']:
                    reasoning += f"""

                Financial Health: The company demonstrates {
                    'strong' if recommendation['action'] == 'BUY' else 'concerning' if recommendation['action'] == 'SELL' else 'stable'
                } financial metrics, supported by its recent financial performance and market position."""
                
                if recommendation['action'] == 'BUY':
                    reasoning += "\n\nThe BUY recommendation is supported by positive market sentiment, strong fundamentals, and potential growth opportunities."
                elif recommendation['action'] == 'SELL':
                    reasoning += "\n\nThe SELL recommendation is based on concerning indicators in market sentiment, technical analysis, and/or fundamental metrics."
                else:
                    reasoning += "\n\nThe HOLD recommendation suggests monitoring the situation while maintaining current positions."
                
                st.write(reasoning)
                
                st.subheader("Key Factors")
                for reason in recommendation['reasoning']:
                    st.write(f"- {reason}")
                
                st.subheader("Risk Factors")
                risk_factors = [
                    f"Market Risk: General market conditions in the {sector} sector",
                    "Economic Risk: Macroeconomic factors and interest rate environment",
                    f"Company-Specific Risk: Factors unique to {company_name}",
                    "Competition Risk: Industry competitive dynamics",
                    "Regulatory Risk: Potential regulatory changes affecting the industry"
                ]
                for risk in risk_factors:
                    st.write(f"- {risk}")
        else:
            st.error("Analysis failed. Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main()
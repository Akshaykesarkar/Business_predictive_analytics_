# Import required libraries
# import requests 
import os
import pandas as pd
import numpy as np
import shap
# import torch
import streamlit as st
# import seaborn as sns
# import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from apikey import apikey
from langchain_community.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from streamlit_chat import message
from sklearn.ensemble import IsolationForest 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import re

# from pandas_profiling import ProfileReport
# import sweetviz as sv
# import gc
# import webbrowser
from streamlit_option_menu import option_menu
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
import plotly.figure_factory as ff
from groq import Groq
import re
import json
from streamlit_lottie import st_lottie


st.set_page_config(layout="wide")

# OpenAI API Key
os.environ['OPENAI_API_KEY'] = apikey
load_dotenv(find_dotenv())

# Streamlit App Title
with open("animation.json", "r", encoding="utf-8") as f:
    lottie_animation = json.load(f)

col1, col2 = st.columns([2, 2])

# Streamlit App Title
with col1:
    st.title('AI-Powered Business Predictive Analytics🤖')
    st.write("Welcome to the AI-Powered Data Science Assistant! This app helps you explore datasets, select machine learning models, and interact with an AI chat assistant for data science queries.")
with col2:
    st_lottie(lottie_animation, height=320, key="stock_animation")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = option_menu(
    menu_title=None,  # No menu title
    options=["Data Cleaning","EDA", "Model Selection", "Chat Assistant"],
    icons=["activity","bar-chart", "code", "chat"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

if 'charts' not in st.session_state:
    st.session_state.charts = []
if 'chart_counter' not in st.session_state:
    st.session_state.chart_counter = 0  # Counter for unique chart keys
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'visualizations_done' not in st.session_state:
    st.session_state.visualizations_done = False

# Load Dataset
uploaded_file = st.sidebar.file_uploader("Upload your dataset(s) (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.session_state.analysis_done = False
    st.session_state.visualizations_done = False
    st.session_state.charts = []  # Clear existing charts
    st.session_state.chart_counter = 0  # Reset counter
    st.sidebar.success("Files uploaded successfully!")

# Initialize Groq client
def initialize_groq_client():
    try:
        client = Groq(api_key="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")
        return client
    except Exception as e:
        st.error(f"An error occurred while initializing Groq client: {str(e)}")
        return None

if app_mode == "Data Cleaning":
    def analyze_dataset_with_groq(df: pd.DataFrame) -> dict:
        """Get cleaning recommendations from Groq's LLM"""
        client = Groq(api_key="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")
        analysis_prompt = f"""
        Analyze this dataset and suggest cleaning steps:
        - Columns: {df.columns.tolist()}
        - Dtypes: {df.dtypes.to_dict()}
        - Missing values: {df.isna().sum().to_dict()}
        - Unique values in string columns: {{
            col: df[col].nunique() 
            for col in df.select_dtypes(include=['object']).columns
        }}

        Follow these rules:
        1. Missing values:
        - If <5% missing, drop rows
        - Else impute (mean/median for numeric, mode for categorical)
        2. String columns:
        - If ≤10 unique values, convert to category
        - If >10 unique values and high cardinality (>50% unique), drop
        3. Remove unnecessary columns (IDs, constants, etc.)
        4. Dont convert date into categorical or dont drop it

        Return JSON with:
        - columns_to_drop
        - columns_to_convert_to_category
        - missing_value_strategy (per column)
        - imputation_method (where applicable)
        """
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": analysis_prompt}],
            model="deepseek-r1-distill-llama-70b",
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=131072
        )
        
        return json.loads(response.choices[0].message.content)

    def clean_dataset_rules(df: pd.DataFrame) -> pd.DataFrame:
        """Rules-based cleaning without LLM"""
        cleaned_df = df.copy()
        
        # 1. Handle missing values
        for col in cleaned_df.columns:
            missing_pct = cleaned_df[col].isna().mean()
            
            if missing_pct > 0:
                if missing_pct < 0.05:
                    cleaned_df = cleaned_df.dropna(subset=[col])
                else:
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
        
        # 2. Handle string columns
        string_cols = cleaned_df.select_dtypes(include=['object']).columns
        for col in string_cols:
            unique_count = cleaned_df[col].nunique()
            
            if unique_count <= 10:
                cleaned_df[col] = cleaned_df[col].astype('category')
            elif unique_count > 10 and unique_count/len(cleaned_df) > 0.5:
                cleaned_df = cleaned_df.drop(columns=[col])
        
        # 3. Remove unnecessary columns
        cleaned_df = cleaned_df.dropna(axis=1, how='all')
        
        for col in cleaned_df.columns:
            if cleaned_df[col].nunique() == 1:
                cleaned_df = cleaned_df.drop(columns=[col])
        
        for col in cleaned_df.columns:
            if cleaned_df[col].nunique()/len(cleaned_df) > 0.9:
                cleaned_df = cleaned_df.drop(columns=[col])
        
        return cleaned_df
    
    st.header("🧹 Smart Data Cleaner")
    st.subheader("Automatically clean your dataset with AI or rules-based logic")
    
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
            
        # Display original data
        st.subheader("Original Data")
        st.write(f"Shape: {df.shape}")
        st.dataframe(df.head())
        
        # Show data stats
        with st.expander("Show Data Statistics"):
            col1,col2 = st.columns(2)
            with col1:
                st.write("Missing Values:")
                st.write(df.isna().sum())
            with col2:
                st.write("\nData Types:")
                st.write(df.dtypes)
        
        # Cleaning options
        st.subheader("Cleaning Options")
        use_ai = st.checkbox("Use AI-powered cleaning (Groq)", value=True)
        
        if st.button("Clean Dataset"):
            with st.spinner("Cleaning in progress..."):
                try:
                    if use_ai:
                        # AI-powered cleaning
                        recommendations = analyze_dataset_with_groq(df)
                        cleaned_df = df.copy()
                        
                        # Apply recommendations
                        for col, strategy in recommendations.get("missing_value_strategy", {}).items():
                            if col in cleaned_df.columns:
                                if strategy == "drop_rows":
                                    cleaned_df = cleaned_df.dropna(subset=[col])
                                elif strategy == "impute":
                                    method = recommendations["imputation_method"].get(col)
                                    if method == "mean":
                                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                                    elif method == "median":
                                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                                    elif method == "mode":
                                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
                        
                        for col in recommendations.get("columns_to_convert_to_category", []):
                            if col in cleaned_df.columns:
                                cleaned_df[col] = cleaned_df[col].astype('category')
                        
                        columns_to_drop = [col for col in recommendations.get("columns_to_drop", []) 
                                        if col in cleaned_df.columns]
                        cleaned_df = cleaned_df.drop(columns=columns_to_drop)
                        
                        st.success("AI-powered cleaning complete!")
                    else:
                        # Rules-based cleaning
                        cleaned_df = clean_dataset_rules(df)
                        st.success("Rules-based cleaning complete!")
                    
                    # Show cleaned data
                    st.subheader("Cleaned Data")
                    st.write(f"New shape: {cleaned_df.shape}")
                    st.dataframe(cleaned_df.head(20))
                    
                    # Show changes
                    with st.expander("Show Cleaning Report"):
                        st.write("Columns removed:", set(df.columns) - set(cleaned_df.columns))
                        st.write("Columns converted to category:", 
                                [col for col in cleaned_df.columns 
                                if pd.api.types.is_categorical_dtype(cleaned_df[col])])
                    
                    # Download cleaned data
                    csv = cleaned_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Cleaned Data",
                        csv,
                        "cleaned_data.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                except Exception as e:
                    st.error(f"Error during cleaning: {str(e)}")


# ---- 1️⃣ Exploratory Data Analysis (EDA) ---- #
if app_mode == "EDA":
    st.header("📊 Exploratory Data Analysis")

    def sanitize_dataframe(df):
        if df is not None:
            numeric_df = df.select_dtypes(include=[np.number])
            return numeric_df

    def summarize_dataframe(df):
        if df is not None:
            summary = df.describe(include='all').to_json(orient='records')
            return summary

    def analyze_data_with_groq(client, df):
        try:
            df_cleaned = sanitize_dataframe(df)
            if df_cleaned.empty:
                return ""

            summary = summarize_dataframe(df_cleaned)
            
            completion = client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this data summary and provide:
                    1. Key Facts (bullet points of essential metrics)
                    2. Recommendations (actionable insights)
                    3. Summarize data anomalies.
                    4. Suggest missing feature imputation techniques.
                    5. Automate business insight extraction
                    6. Conclusion (final observation)
                
                    Summary: {summary}"""
                }],
                temperature=0.4,
                max_tokens=131072,
                top_p=0.9
            )

            raw_output = completion.choices[0].message.content
            cleaned_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)
            
            return cleaned_output.strip()

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return ""

    def visualize_data_with_groq(client, df):
        try:
            df_cleaned = sanitize_dataframe(df)
            if df_cleaned.empty:
                return

            # Enhanced prompt with strict dataset usage instructions
            prompt = f"""Generate visualizations using THE ACTUAL DATASET PROVIDED (df_cleaned).
            DO NOT CREATE OR MODIFY THE DATASET. Use these columns: {list(df_cleaned.columns)}
            
            Requirements:
            1. Use ONLY this data: df_cleaned (shape: {df_cleaned.shape})
            Convert category columns into numerical values and then visualize
            2. Forbidden:
            - Any pd.DataFrame() creations
            - Hardcoded data
            - Example/test data
            - Box plots
            - Exclude ID-like columns: {['Unnamed: 0', 'Order ID', 'Pizza ID']}
            - Unwanted graphs
            3. Required visualizations:
            - Temporal trends (line/area charts)
            - Correlation analysis of Columns (using One scatter plots or heatmap is must)
            - Distribution analysis (histograms)
            - Categorical breakdowns (bar/pie)
            - Hourly patterns (heatmaps)
            - Include many other visualization like geographic when longitude and latitude is present (use st.map(map_data))
            - Use Most of the columns
            4. Create 8 different chart/plots types focusing on these relationships
            5. OUTPUT FORMAT:
            - Only Python code within ```python blocks
            - One visualization per code block
            - Include necessary aggregations
            6. Each visualization MUST:
                * Use Plotly Express
                * Have meaningful title starting with "Fig [N]: "
                * Include axis labels with units
                * Represents columns using colors in ONLY scatter plots
                * Contain <50 words caption in # comments explaining insight
                * Use st.plotly_chart() with full width
            
            Example VALID code:
            ```python
            # Fig 1: Sales distribution by month
            monthly_sales = df_cleaned.groupby('Month', as_index=False)['Sales'].sum()
            fig = px.bar(monthly_sales, x='Month', y='Sales', 
                        title='Actual Monthly Sales from Dataset')
            st.plotly_chart(fig, use_container_width=True)
            ```
            
            First 3 rows of ACTUAL DATA:
            {df_cleaned.head(3).to_string()}
            """

            completion = client.chat.completions.create(
                model="qwen-2.5-coder-32b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,  # Lower temperature for less creativity
                max_tokens=131072
            )

            response = completion.choices[0].message.content
            code_blocks = re.findall(r'```python(.*?)```', response, re.DOTALL)

            if not code_blocks:
                st.error("No valid code found in response")
                return

            exec_globals = {
                'pd': pd, 'np': np, 'st': st,
                'px': px, 'go': go, 'ff': ff,
                'df_cleaned': df_cleaned  # Pass actual dataframe
            }

            st.subheader("Full Dataset Visualizations")
            for idx, code in enumerate(code_blocks, 1):
                try:
                    # Validate code contains actual dataset reference
                    if 'df_cleaned' not in code:
                        st.error(f"Visualization {idx} rejected: No dataset reference")
                        continue
                        
                    if 'pd.DataFrame(' in code:
                        st.error(f"Visualization {idx} rejected: Creates new dataframe")
                        continue

                    with st.expander(f"Visualization {idx}: Code", expanded=False):
                        st.code(code.strip(), language='python')
                    
                    # Execute in controlled environment
                    exec(code.strip(), exec_globals)
                    
                    # # Force display if figure wasn't shown
                    # if 'fig' in exec_globals:
                    #     st.plotly_chart(exec_globals['fig'], use_container_width=True)

                except Exception as e:
                    st.error(f"Error in visualization {idx}: {str(e)}")

        except Exception as e:
            st.error(f"Visualization error: {str(e)}")

    # Main execution flow
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        st.write("Data Preview:")
        st.write(df.head())
        
        client = initialize_groq_client()
        if client:
            # Analysis section
            with st.expander("Data Analysis", expanded=True):
                analysis_result = analyze_data_with_groq(client, df)
                if analysis_result:
                    st.markdown(analysis_result)
                st.session_state.analysis_done = True
            
            # Visualization section
            visualize_data_with_groq(client, df)
            st.session_state.visualizations_done = True

            
            # Query section - appears after initial analysis/visualization
            st.divider()
            st.subheader("Custom Query Visualization")
            
            query = st.text_input("Enter your query for visualization (e.g., 'Show sales by category')")
            
            if st.button("Generate Visualization") and query:
                df_cleaned = sanitize_dataframe(df)
                if df_cleaned.empty:
                    st.error("No numeric data available for visualization")
                else:
                    prompt = f"""Generate a visualization based on this query:
                    Query: {query}
                    
                    Dataset Columns: {list(df_cleaned.columns)}
                    First 3 rows:
                    {df_cleaned.head(3).to_string()}
                    
                    Requirements:
                    1. Use ONLY df_cleaned
                    2. Output Plotly code in ```python block
                    3. Must use st.plotly_chart()"""
                    
                    completion = client.chat.completions.create(
                        model="deepseek-r1-distill-llama-70b",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                        max_tokens=131072
                    )
                    
                    response = completion.choices[0].message.content
                    code_blocks = re.findall(r'```python(.*?)```', response, re.DOTALL)
                    
                    if code_blocks:
                        exec_globals = {
                            'pd': pd, 'np': np, 'st': st,
                            'px': px, 'go': go, 'ff': ff,
                            'df_cleaned': df_cleaned
                        }
                        
                        try:
                            code = code_blocks[0].strip()
                            if 'df_cleaned' in code and 'pd.DataFrame(' not in code:
                                with st.expander("Generated Code", expanded=False):
                                    st.code(code, language='python')
                                
                                exec(code, exec_globals)
                                
                                if 'fig' in exec_globals:
                                    # Validate the figure
                                    if hasattr(exec_globals['fig'], '_grid_ref') or hasattr(exec_globals['fig'], 'data'):
                                        chart_key = f"query_chart_{st.session_state.chart_counter}"
                                        st.session_state.chart_counter += 1
                                        
                                        st.session_state.charts.append({
                                            'fig_object': exec_globals['fig'],
                                            'key': chart_key,
                                            'type': 'query'
                                        })
                                        
                                        st.plotly_chart(
                                            exec_globals['fig'],
                                            key=chart_key,
                                            use_container_width=True
                                        )
                                        st.success("Visualization added to dashboard!")
                                    else:
                                        st.error("Generated figure is invalid")
                        except Exception as e:
                            st.error(f"Error executing generated code: {str(e)}")
                    else:
                        st.error("No valid code found in response")

        # Dashboard display
        if st.session_state.charts:
            st.divider()
            st.subheader("Dashboard")
            for chart_data in st.session_state.charts:
                try:
                    if 'fig_object' in chart_data:
                        st.plotly_chart(
                            chart_data['fig_object'],
                            key=chart_data['key'],
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error displaying chart: {str(e)}")
                
                
if app_mode == "Model Selection":
    st.header("🧠 Context-Aware Model Selection (Hybrid Intelligence)")

    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        # Automated Problem Categorization
        st.subheader("🔍 Problem Identification")
        
        # Rule-based problem detection
        date_columns = [col for col in df.columns 
                       if re.search(r'date|time|year|month|day', col, re.I)]
        target_var = st.selectbox("Select Target Variable", df.columns)
        
        # Heuristic rules from research paper
        problem_type = "classification"
        if len(date_columns) > 0:
            problem_type = "time-series"
        elif pd.api.types.is_numeric_dtype(df[target_var]):
            problem_type = "regression" if df[target_var].nunique() > 10 else "classification"
        
        # LLM-assisted validation
        client = initialize_groq_client()
        llm_check = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": f"""Confirm problem type {problem_type} 
                      for target '{target_var}' with features {df.columns}. Answer yes/no"""}],
            temperature=0.3,
            max_tokens=10
        ).choices[0].message.content
        
        if "no" in llm_check.lower():
            problem_type = st.selectbox("Select Problem Type", 
                                      ["time-series", "regression", "classification"])
        
        st.success(f"Identified Problem Type: {problem_type.upper()}")

        # Temporal Feature Engineering (Fixed Version)
        if problem_type == "time-series" and date_columns:
            date_col = date_columns[0]
            
            # Convert to datetime with error handling
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Drop invalid dates
            df = df.dropna(subset=[date_col])
            
            # Set and sort datetime index
            df = df.set_index(date_col).sort_index()
            
            # Resample with forward filling
            window_size = st.slider("Historical Window Size (days)", 7, 60, 30)
            
            # Modified resampling with proper column handling
            df = (df.resample('D')
                  .mean(numeric_only=True)
                  .ffill()
                  .reset_index()
                  .rename(columns={'index': date_col})  # Ensure consistent column naming
                  .set_index(date_col)
                  .sort_index())
            
            # Create rolling features
            df['rolling_mean'] = df[target_var].rolling(window=window_size, min_periods=1).mean()

        # Rest of the model selection code remains the same...
            df = df.dropna()

        # Automated Model Selection
        model = None
        if problem_type == "time-series":
            model = "LSTM"
            params = {'units': 50, 'window': 30, 'epochs': 100}
        elif problem_type == "regression":
            model = "XGBoost"
            params = {'n_estimators': 100, 'max_depth': 3}
        else:
            model = "Random Forest"
            params = {'n_estimators': 100, 'max_depth': 5}
        
        st.subheader(f"⚙️ Auto-Selected Model: {model}")
        
        if st.button("Train & Predict"):
            if problem_type == "time-series":
                # LSTM with Daily Averages
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df[[target_var]])
                
                # Create sequences
                X, y = [], []
                for i in range(params['window'], len(scaled_data)):
                    X.append(scaled_data[i-params['window']:i])
                    y.append(scaled_data[i])
                X, y = np.array(X), np.array(y)
                
                # Build model
                lstm_model = Sequential()
                lstm_model.add(LSTM(params['units'], input_shape=(X.shape[1], 1)))
                lstm_model.add(Dense(1))
                lstm_model.compile(optimizer='adam', loss='mse')
                lstm_model.fit(X, y, epochs=params['epochs'], verbose=0)
                
                # Generate 20-day forecast
                last_seq = scaled_data[-params['window']:]
                predictions = []
                for _ in range(30):
                    pred = lstm_model.predict(last_seq.reshape(1, params['window'], 1))
                    predictions.append(pred[0,0])
                    last_seq = np.append(last_seq[1:], pred)
                
                # Inverse transform
                predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))

                # Create forecast dates
                last_date = df.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),  # Fixed datetime math
                    periods=30
                )

                # Plot forecast-only graph
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=predictions.flatten(),
                    mode='lines+markers',
                    name='30-Day Forecast',
                    line=dict(color='#FF6B6B', width=3)
                ))
                fig.update_layout(
                    title=f'30-Day {target_var} Forecast',
                    xaxis_title='Date',
                    yaxis_title=target_var,
                    showlegend=True
                )
                st.plotly_chart(fig)

                # LLM Analysis with Clean Insights
                client = initialize_groq_client()
                if client:
                    analysis_prompt = f"""Analyze this forecast pattern and provide:
                    1. Key trend observations (upward/downward/sideways)
                    2. Notable fluctuations in next 30 days
                    3. Business recommendations based on predictions
                    4. Risk factors to monitor
                    
                    Data:
                    - First 5 forecast values: {predictions.flatten()[:5].round(2).tolist()}
                    - Last 5 forecast values: {predictions.flatten()[-5:].round(2).tolist()}
                    - Maximum predicted value: {round(predictions.max(), 2)}
                    - Minimum predicted value: {round(predictions.min(), 2)}
                    
                    Format: Clean bullet points without any XML tags."""
                    
                    try:
                        raw_analysis = client.chat.completions.create(
                            model="deepseek-r1-distill-llama-70b",
                            messages=[{"role": "user", "content": analysis_prompt}],
                            temperature=0.4,
                            max_tokens=512
                        ).choices[0].message.content

                        # Clean output
                        clean_analysis = re.sub(r'<[^>]+>', '', raw_analysis)
                        
                        st.subheader("🔍 Forecast Insights")
                        st.markdown(f"```\n{clean_analysis}\n```")

                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

                
            else:
                # Feature Analysis
                selected_feature = st.selectbox("Analyze Feature Impact", 
                                               df.drop(columns=[target_var]).columns)
                feature_range = st.slider(f"{selected_feature} Range",
                                         float(df[selected_feature].min()),
                                         float(df[selected_feature].max()),
                                         (float(df[selected_feature].quantile(0.25)),
                                         float(df[selected_feature].quantile(0.75))))
                
                # Regression/Classification Model
                X = df.drop(columns=[target_var])
                y = df[target_var]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                if problem_type == "regression":
                    model = XGBRegressor(**params)
                    model.fit(X_train, y_train)
                    test_values = np.linspace(feature_range[0], feature_range[1], 20)
                    predictions = []
                    for val in test_values:
                        sample = X.mean().to_dict()
                        sample[selected_feature] = val
                        predictions.append(model.predict([list(sample.values())])[0])
                else:
                    model = RandomForestClassifier(**params)
                    model.fit(X_train, y_train)
                    test_values = np.linspace(feature_range[0], feature_range[1], 20)
                    predictions = []
                    for val in test_values:
                        sample = X.mean().to_dict()
                        sample[selected_feature] = val
                        predictions.append(model.predict_proba([list(sample.values())])[0][1])
                
                # Plot feature impact
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test_values, y=predictions))
                fig.update_layout(title=f"{selected_feature} Impact on {target_var}",
                                xaxis_title=selected_feature,
                                yaxis_title="Prediction")
                st.plotly_chart(fig)
    else:
        st.warning("Please upload dataset first")

# ---- 3️⃣ AI Chat Assistant ---- #
if app_mode == "Chat Assistant":
    st.header("💬 AI-Powered Chat Assistant")

    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["Hello! How can I assist you?"]
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    # Initialize the LLM if not in session state
    if 'llm' not in st.session_state:
        st.session_state.llm = OpenAI(temperature=0)

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=st.session_state.llm, verbose=True)

    # Load dataset into the agent if available
    if df is not None:
        pandas_agent = create_pandas_dataframe_agent(st.session_state.llm, df, verbose=True, allow_dangerous_code=True)
    else:
        pandas_agent = None

    with st.container():
        query = st.text_input("Ask me anything about Data Science or the uploaded dataset! 🤖")

        if query:
            with st.spinner("Thinking..."):
                # Check if the query is related to the dataset
                dataset_keywords = ["column", "row", "mean", "sum", "max", "min", "missing values", "outliers", "statistics"]
                
                if pandas_agent and any(keyword in query.lower() for keyword in dataset_keywords):
                    response = pandas_agent.run(query)
                else:
                    response = conversation.predict(input=query)

            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

    # Clear chat option
    if st.button("Clear Chat"):
        st.session_state['responses'] = ["Hello! How can I assist you?"]
        st.session_state['requests'] = []
        st.success("Chat cleared!")

    # Display conversation history
    with st.container():
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i], key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

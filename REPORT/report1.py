import streamlit as st
import pandas as pd
import openai
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load environment variables securely
openai.api_key = ""

def generate_summary(dataframe):
    text = dataframe.to_csv(index=False)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful and kind assistant."},
            {"role": "user", "content": f"Summarize the following data:\n{text}"}
        ]
    )
    return response.choices[0].message['content']

def query_data(dataframe, user_query):
    text = dataframe.to_csv(index=False)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{user_query}\nHere is the data:\n{text}"}
        ]
    )
    return response.choices[0].message['content']

def plot_bar_chart(dataframe, column):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=dataframe, x=column, order=dataframe[column].value_counts().index)
    plt.title(f'Count of Each {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(plt)

def plot_line_chart(dataframe, column):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=dataframe, x=dataframe.index, y=column)
    plt.title(f'Line Chart of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    st.pyplot(plt)

def plot_pie_chart(dataframe, column):
    plt.figure(figsize=(8, 8))
    dataframe[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f'Pie Chart of {column}')
    st.pyplot(plt)

def plot_histogram(dataframe, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dataframe, x=column, bins=20, kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    st.pyplot(plt)

def interpret_visualization_prompt(prompt, dataframe):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that helps interpret data visualization requests."},
            {"role": "user", "content": f"Extract the chart type and relevant column names from the following prompt. Respond in JSON format with keys 'chart_type' and 'columns'. Available chart types are: Bar Chart, Line Chart, Pie Chart, Histogram.\n\nPrompt: {prompt}"}
        ]
    )
    try:
        content = response.choices[0].message['content']
        parsed = json.loads(content)
        chart_type = parsed.get("chart_type")
        columns = parsed.get("columns")
        if isinstance(columns, str):
            columns = [columns]
        return chart_type, columns
    except json.JSONDecodeError:
        st.error("Failed to parse the visualization prompt. Please try rephrasing your request.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None

def load_data(file):
    """Load data based on the file type."""
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    elif file.name.endswith('.txt'):
        return pd.read_csv(file, delimiter="\t")  # assuming tab-delimited text file
    else:
        st.error("Unsupported file format. Please upload a CSV, Excel, JSON, or text file.")
        return None

# Streamlit UI
st.title("Data Analysis and Visualization")

menu = st.sidebar.selectbox("Choose an Option", ["Summarize", "Query Data", "Visualize Data"])

if menu == "Summarize":
    st.subheader("Summarization of Your Data")
    file_uploader = st.file_uploader("Upload your data file", type=["csv", "xlsx", "json", "txt"])
    if file_uploader is not None:
        df = load_data(file_uploader)
        if df is not None:
            st.write("Data Loaded Successfully:")
            st.write(df)
            summary = generate_summary(df)
            st.write("Summary:")
            st.write(summary)

elif menu == "Query Data":
    st.subheader("Query Your Data")
    file_uploader = st.file_uploader("Upload your data file", type=["csv", "xlsx", "json", "txt"])
    if file_uploader is not None:
        df = load_data(file_uploader)
        if df is not None:
            st.write("Data Loaded Successfully:")
            st.write(df)
            user_query = st.text_input("Ask a question about your data:")
            if st.button("Get Answer"):
                if user_query:
                    answer = query_data(df, user_query)
                    st.write("Answer:")
                    st.write(answer)

elif menu == "Visualize Data":
    st.subheader("Visualize Your Data")
    file_uploader = st.file_uploader("Upload your data file", type=["csv", "xlsx", "json", "txt"])
    if file_uploader is not None:
        df = load_data(file_uploader)
        if df is not None:
            st.write("Data Loaded Successfully:")
            st.write(df)
            
            visualization_option = st.radio("Choose Visualization Method", ["Select Chart Type", "Prompt-Based"])
            
            if visualization_option == "Select Chart Type":
                column = st.selectbox("Select a column to visualize", df.columns)
                chart_type = st.selectbox("Select the type of chart", ["Bar Chart", "Line Chart", "Pie Chart", "Histogram"])
                
                if st.button("Generate Visualization"):
                    if chart_type == "Bar Chart":
                        plot_bar_chart(df, column)
                    elif chart_type == "Line Chart":
                        plot_line_chart(df, column)
                    elif chart_type == "Pie Chart":
                        plot_pie_chart(df, column)
                    elif chart_type == "Histogram":
                        plot_histogram(df, column)
            
            elif visualization_option == "Prompt-Based":
                prompt = st.text_area("Describe the visualization you want:")
                if st.button("Generate Visualization"):
                    if prompt:
                        chart_type, columns = interpret_visualization_prompt(prompt, df)
                        if chart_type and columns:
                            # Validate chart type
                            valid_chart_types = ["Bar Chart", "Line Chart", "Pie Chart", "Histogram"]
                            if chart_type not in valid_chart_types:
                                st.error(f"Unsupported chart type: {chart_type}. Please choose from {valid_chart_types}.")
                            else:
                                # Validate columns
                                missing_columns = [col for col in columns if col not in df.columns]
                                if missing_columns:
                                    st.error(f"Columns not found in data: {missing_columns}")
                                else:
                                    # For simplicity, handle single column visualizations
                                    if len(columns) == 1:
                                        column = columns[0]
                                        if chart_type == "Bar Chart":
                                            plot_bar_chart(df, column)
                                        elif chart_type == "Line Chart":
                                            plot_line_chart(df, column)
                                        elif chart_type == "Pie Chart":
                                            plot_pie_chart(df, column)
                                        elif chart_type == "Histogram":
                                            plot_histogram(df, column)
                                    else:
                                        st.error("Currently, only single-column visualizations are supported.")

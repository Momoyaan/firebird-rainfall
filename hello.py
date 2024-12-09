import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Rainfall Data Analysis")
with open("styles.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        [data-testid=stSidebar] {
            border-right: 1px solid #39393B !important;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Load and preprocess the data
@st.cache_data(persist=True)
def load_data():
    df = pd.read_csv("rainfall.csv")
    df = df.dropna()

    numerical_columns = ["rainfall", "temperature", "humidity", "wind_speed"]
    categorical_columns = ["weather_condition"]

    num_imputer = SimpleImputer(strategy="mean")
    df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numerical_columns])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = kmeans.fit_predict(df_scaled)

    return df


# Main Streamlit app
def main():
    st.title("Rainfall Analysis")

    df = load_data()

    # Sidebar navigation
    pages = {
        "Data Summary": show_data_summary,
        "Clustering": show_clustering_plot,
        "Linear Regression": show_linear_regression,
        "Linear Regression: Rainfall vs Temperature": show_rainfall_vs_temperature,
        "Linear Reggression: Rainfall vs Humidity": show_rainfall_vs_humidity,
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    pages[selection](df)


# Show data summary
def show_data_summary(df):
    st.markdown("""
This Streamlit app is designed to analyze rainfall data. The data includes various factors such as temperature, humidity, wind speed, and weather conditions. The app uses clustering and linear regression techniques to gain insights into the relationships between these factors and rainfall.
""")
    st.subheader("Data Summary")
    st.markdown("""
    The data is preprocessed by removing missing values, imputing numerical columns with the mean, and imputing categorical columns with the most frequent value. The data is then scaled using StandardScaler to have a mean of 0 and a standard deviation of 1.
    """)
    st.write(df.describe())
    st.markdown("""
    The data summary provides an overview of the central tendency and dispersion of each variable. It includes the count, mean, standard deviation, minimum, 25th percentile, median, 75th percentile, and maximum values for each variable.
    """)


# Show scatter plot of temperature vs rainfall colored by cluster
def show_clustering_plot(df):
    st.subheader("Temperature vs Rainfall (colored by cluster)")
    fig = px.scatter(
        df,
        x="temperature",
        y="rainfall",
        color="cluster",
        color_continuous_scale="plasma",
    )
    st.plotly_chart(fig)
    st.markdown("""
    The clustering plot shows the relationship between temperature and rainfall, colored by cluster. The clusters are formed using K-means clustering algorithm, which groups similar data points into clusters.
    """)


# Show linear regression plots
def show_linear_regression(df):
    st.subheader("Linear Regression")

    X = df[["temperature", "humidity"]]
    y = df["rainfall"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-squared (R²): {r2:.2f}")
    st.markdown("""
    The linear regression model predicts the rainfall based on temperature and humidity. The mean squared error (MSE) measures the average difference between predicted and actual values, while the R-squared (R²) measures the proportion of variance in the dependent variable that is predictable from the independent variable(s).
    """)

    # Show actual vs predicted rainfall plot
    fig = px.scatter(x=y_test, y=y_pred, color_discrete_sequence=["#9b59b6"])
    fig.add_shape(
        type="line",
        x0=y_test.min(),
        x1=y_test.max(),
        y0=y_test.min(),
        y1=y_test.max(),
        line=dict(color="#e74c3c", width=2),
    )
    fig.update_layout(
        title="Actual vs Predicted Rainfall",
        xaxis_title="Actual Rainfall",
        yaxis_title="Predicted Rainfall",
    )
    st.plotly_chart(fig)
    st.markdown("""
    The actual vs predicted rainfall plot shows the relationship between actual and predicted values. The line of perfect prediction is shown in red.
    """)


# Show rainfall vs temperature plot
def show_rainfall_vs_temperature(df):
    st.subheader("Rainfall vs Temperature")
    fig = px.scatter(
        df,
        x="temperature",
        y="rainfall",
        trendline="ols",
        trendline_color_override="#e67e22",
    )
    fig.update_traces(marker=dict(color="#2ecc71"))
    st.plotly_chart(fig)
    st.markdown("""
    The rainfall vs temperature plot shows the relationship between rainfall and temperature. The trendline is shown in orange.
    """)


# Show rainfall vs humidity plot
def show_rainfall_vs_humidity(df):
    st.subheader("Rainfall vs Humidity")
    fig = px.scatter(
        df,
        x="humidity",
        y="rainfall",
        trendline="ols",
        trendline_color_override="#9b59b6",
    )
    fig.update_traces(marker=dict(color="#f1c40f"))
    st.plotly_chart(fig)
    st.markdown("""
    The rainfall vs humidity plot shows the relationship between rainfall and humidity. The trendline is shown in purple.
    """)


if __name__ == "__main__":
    main()

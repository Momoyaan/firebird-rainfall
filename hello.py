import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
    df["date"] = pd.to_datetime(df["date"])

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
        "Linear Regression: Rainfall vs Humidity": show_rainfall_vs_humidity,
        "Overall Insights": show_overall_insights,
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    pages[selection](df)


# Show data summary
def show_data_summary(df):
    st.markdown("""
    This Streamlit app is designed to analyze rainfall data. The data includes various factors such as temperature, humidity, wind speed, and weather conditions. The app uses k-means clustering and linear regression techniques to gain insights into the relationships between these factors and rainfall.
    """)

    st.subheader("Data Summary")

    st.markdown("""
    The data is preprocessed by performing the following steps:
    1. Removing missing values (`dropna`).
    2. Imputing numerical columns with the mean value.
    3. Imputing categorical columns with the most frequent value.
    4. Scaling numerical features for better model performance.

    Below is the overview of missing values before and after data cleaning.
    """)

    # Manually create missing data table before cleaning
    st.subheader("Missing Data Before Cleaning")
    missing_before = {
        "date": 0,
        "rainfall": 1,
        "temperature": 1,
        "humidity": 1,
        "wind_speed": 1,
        "weather_condition": 1,
    }

    missing_df = pd.DataFrame(
        list(missing_before.items()), columns=["Column", "Missing Data"]
    )
    st.write(missing_df)

    # Proceed with data cleaning steps
    numerical_columns = ["rainfall", "temperature", "humidity", "wind_speed"]
    categorical_columns = ["weather_condition"]

    # Impute numerical columns with the mean
    num_imputer = SimpleImputer(strategy="mean")
    df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])

    # Impute categorical columns with the most frequent value
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

    # Drop any remaining missing values
    df = df.dropna()

    # Show missing data after cleaning (should be all zeros)
    st.subheader("Missing Data After Cleaning")
    missing_after = df.isnull().sum()
    st.write(missing_after)

    st.markdown("""
    After data cleaning, all missing values were handled successfully. The missing data is now zero for all columns.
    """)

    # Display data summary (descriptive statistics)
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
    The scatter plot shows the relationship between temperature and rainfall where the x-axis represents the temperature, and the y-axis for rainfall, colored by cluster. The clusters are formed using K-means clustering algorithm, which groups similar data points into clusters.
    """)
    st.markdown("""
    ### Cluster Observations:
    - **Cluster 0 (Blue)**: Mostly consists of data points where:
      - Rainfall is relatively higher (14.8 to 21.8 units).
      - Temperature is on the lower side (around 13.9 to 17.1).
    - **Cluster 1 (Pink)**: Includes data points with:
      - Minimal rainfall (equal to 2.1).
      - Higher temperatures (19.4 to 23.4).
    - **Cluster 2 (Yellow)**: Features data points where:
      - Rainfall ranges from 3.9 to 11.6 units.
      - Temperature ranges between 16.1 and 19.7.
    """)

    # Elbow Method to find the optimal number of clusters
    st.subheader("Elbow Method for Optimal k")
    numerical_columns = ["rainfall", "temperature", "humidity", "wind_speed"]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numerical_columns])

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)

    # Create a Plotly figure for the Elbow Method
    fig = px.line(
        x=range(1, 11), y=wcss, labels={"x": "Number of Clusters", "y": "WCSS"}
    )
    fig.update_layout(
        title="Elbow Method for Optimal k",
        xaxis_title="Number of clusters",
        yaxis_title="WCSS",
        template="plotly_dark",
    )
    st.plotly_chart(fig)

    st.markdown("""
    The **Elbow Method** helps determine the optimal number of clusters by plotting the **WCSS (Within-Cluster Sum of Squares)** for different values of **k** (number of clusters). We can see that the "elbow" is around **k = 3**, meaning adding more clusters beyond 3 doesn’t significantly reduce WCSS.

    Therefore, based on the elbow method, we determined that the optimal number of clusters for this dataset is **3**.
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


def show_rainfall_vs_temperature(df):
    st.header("Rainfall vs Temperature Analysis 🌡️")

    # Add loading animation
    with st.spinner("Loading analysis..."):
        time.sleep(1)  # Simulate loading

    # Interactive date range selector
    st.sidebar.subheader("Filter Data")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [df["date"].min(), df["date"].max()],
        min_value=df["date"].min(),
        max_value=df["date"].max(),
    )

    # Filter data based on date range
    mask = (df["date"] >= pd.to_datetime(date_range[0])) & (
        df["date"] <= pd.to_datetime(date_range[1])
    )
    filtered_df = df[mask]

    # Progress bar for data loading
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    st.success("Data loaded successfully!")

    # Create tabs with animation effect
    tab1, tab2, tab3 = st.tabs(
        ["📈 Interactive Visualization", "📊 Dynamic Analysis", "💡 Real-time Insights"]
    )

    with tab1:
        st.subheader("Temperature-Rainfall Relationship")

        # Interactive metrics with animation
        col1, col2, col3 = st.columns(3)
        with col1:
            current_temp = filtered_df["temperature"].mean()
            prev_temp = df["temperature"].mean()
            st.metric(
                label="Average Temperature",
                value=f"{current_temp:.1f}°C",
                delta=f"{current_temp - prev_temp:.1f}°C from overall average",
                delta_color="inverse",
            )

        # Add interactive features for the plot
        plot_type = st.radio(
            "Select Plot Type", ["Scatter", "Bubble", "3D"], horizontal=True
        )

        if plot_type == "Scatter":
            fig = px.scatter(
                filtered_df,
                x="temperature",
                y="rainfall",
                trendline="ols",
                animation_frame=filtered_df["date"].dt.month,
                range_x=[
                    filtered_df["temperature"].min(),
                    filtered_df["temperature"].max(),
                ],
                range_y=[filtered_df["rainfall"].min(), filtered_df["rainfall"].max()],
                title="Temperature vs Rainfall (Monthly Animation)",
            )

        elif plot_type == "Bubble":
            fig = px.scatter(
                filtered_df,
                x="temperature",
                y="rainfall",
                size="humidity",
                color="wind_speed",
                hover_name="date",
                title="Temperature vs Rainfall (with Humidity and Wind Speed)",
            )

        else:  # 3D plot
            fig = px.scatter_3d(
                filtered_df,
                x="temperature",
                y="humidity",
                z="rainfall",
                color="rainfall",
                title="3D Weather Relationship",
            )

        st.plotly_chart(fig, use_container_width=True)

        # Interactive pattern explorer
        st.subheader("Pattern Explorer")
        selected_temp_range = st.slider(
            "Explore temperature range",
            float(filtered_df["temperature"].min()),
            float(filtered_df["temperature"].max()),
            (
                float(filtered_df["temperature"].min()),
                float(filtered_df["temperature"].max()),
            ),
        )

        # Animated analysis for selected range
        temp_mask = (filtered_df["temperature"] >= selected_temp_range[0]) & (
            filtered_df["temperature"] <= selected_temp_range[1]
        )

        with st.expander("View Detailed Statistics for Selected Range"):
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric(
                    "Average Rainfall",
                    f"{filtered_df[temp_mask]['rainfall'].mean():.2f}mm",
                    f"{filtered_df[temp_mask]['rainfall'].std():.2f}mm std",
                )
            with stats_col2:
                st.metric(
                    "Sample Size",
                    len(filtered_df[temp_mask]),
                    f"{(len(filtered_df[temp_mask])/len(filtered_df))*100:.1f}% of total",
                )

    with tab2:
        st.subheader("Dynamic Analysis")

        # Interactive time series analysis
        fig2 = make_subplots(rows=2, cols=1)
        fig2.add_trace(
            go.Scatter(
                x=filtered_df["date"],
                y=filtered_df["temperature"],
                name="Temperature",
                line=dict(color="red"),
            ),
            row=1,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=filtered_df["date"],
                y=filtered_df["rainfall"],
                name="Rainfall",
                line=dict(color="blue"),
            ),
            row=2,
            col=1,
        )
        fig2.update_layout(
            height=600, title_text="Temperature and Rainfall Time Series"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Add animated correlation matrix
        if st.button("Show Correlation Matrix"):
            with st.spinner("Generating correlation matrix..."):
                time.sleep(1)
                corr_matrix = filtered_df[
                    ["temperature", "rainfall", "humidity", "wind_speed"]
                ].corr()
                fig3 = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    color_continuous_scale="RdBu",
                )
                st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.subheader("Real-time Insights")

        # Interactive threshold analysis
        threshold_temp = st.slider(
            "Temperature Threshold (°C)",
            float(filtered_df["temperature"].min()),
            float(filtered_df["temperature"].max()),
            float(filtered_df["temperature"].mean()),
        )

        # Animated results
        with st.spinner("Analyzing patterns..."):
            time.sleep(0.5)
            above_threshold = filtered_df[filtered_df["temperature"] > threshold_temp][
                "rainfall"
            ].mean()
            below_threshold = filtered_df[filtered_df["temperature"] <= threshold_temp][
                "rainfall"
            ].mean()

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    f"Avg Rainfall Above {threshold_temp:.1f}°C",
                    f"{above_threshold:.2f}mm",
                    f"{above_threshold - below_threshold:.2f}mm difference",
                )
            with col2:
                st.metric(
                    f"Avg Rainfall Below {threshold_temp:.1f}°C",
                    f"{below_threshold:.2f}mm",
                    f"{below_threshold - above_threshold:.2f}mm difference",
                )

        # Add dynamic recommendations
        st.subheader("Weather Insights")
        recommendations = [
            f"Temperature range {selected_temp_range[0]:.1f}°C to {selected_temp_range[1]:.1f}°C shows {filtered_df[temp_mask]['rainfall'].mean():.1f}mm average rainfall",
            f"Strongest rainfall patterns observed at {filtered_df.loc[filtered_df['rainfall'].idxmax(), 'temperature']:.1f}°C",
            f"Current analysis based on {len(filtered_df)} data points",
        ]

        for i, rec in enumerate(recommendations):
            with st.spinner(f"Loading insight {i+1}..."):
                time.sleep(0.3)
                st.info(rec)

        # Add download button with animation
        if st.download_button(
            label="Download Analysis Report",
            data=filtered_df.to_csv().encode("utf-8"),
            file_name="weather_analysis.csv",
            mime="text/csv",
        ):
            st.balloons()


# Show rainfall vs humidity plot
def show_rainfall_vs_humidity(df):
    st.header("Rainfall vs Humidity Analysis 💧")

    # Add loading animation
    with st.spinner("Preparing humidity analysis..."):
        time.sleep(1)

    # Interactive date range selector in sidebar
    st.sidebar.subheader("Data Filter Options")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [df["date"].min(), df["date"].max()],
        min_value=df["date"].min(),
        max_value=df["date"].max(),
    )

    # Filter data based on date range
    mask = (df["date"] >= pd.to_datetime(date_range[0])) & (
        df["date"] <= pd.to_datetime(date_range[1])
    )
    filtered_df = df[mask]

    # Progress bar for data loading
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    st.success("Humidity analysis data loaded successfully!")

    # Create tabs with animation effect
    tab1, tab2, tab3 = st.tabs(
        ["📈 Interactive Plots", "📊 Pattern Analysis", "💡 Humidity Insights"]
    )

    with tab1:
        st.subheader("Humidity-Rainfall Relationship")

        # Interactive metrics with animation
        col1, col2, col3 = st.columns(3)
        with col1:
            current_humidity = filtered_df["humidity"].mean()
            prev_humidity = df["humidity"].mean()
            st.metric(
                label="Average Humidity",
                value=f"{current_humidity:.1f}%",
                delta=f"{current_humidity - prev_humidity:.1f}% from overall",
                delta_color="normal",
            )
        with col2:
            current_rain = filtered_df["rainfall"].mean()
            prev_rain = df["rainfall"].mean()
            st.metric(
                label="Average Rainfall",
                value=f"{current_rain:.1f}mm",
                delta=f"{current_rain - prev_rain:.1f}mm from overall",
                delta_color="normal",
            )
        with col3:
            correlation = filtered_df["rainfall"].corr(filtered_df["humidity"])
            st.metric(
                label="Correlation",
                value=f"{correlation:.2f}",
                delta="Strong positive" if correlation > 0.5 else "Moderate",
            )

        # Interactive plot type selector
        plot_type = st.radio(
            "Select Visualization Type",
            ["Dynamic Scatter", "Bubble Plot", "3D Relationship", "Contour Plot"],
            horizontal=True,
        )

        if plot_type == "Dynamic Scatter":
            fig = px.scatter(
                filtered_df,
                x="humidity",
                y="rainfall",
                trendline="ols",
                animation_frame=filtered_df["date"].dt.month,
                range_x=[filtered_df["humidity"].min(), filtered_df["humidity"].max()],
                range_y=[filtered_df["rainfall"].min(), filtered_df["rainfall"].max()],
                title="Humidity vs Rainfall (Monthly Animation)",
                color="temperature",
                color_continuous_scale="viridis",
            )

        elif plot_type == "Bubble Plot":
            fig = px.scatter(
                filtered_df,
                x="humidity",
                y="rainfall",
                size="wind_speed",
                color="temperature",
                hover_name="date",
                title="Humidity vs Rainfall (with Temperature and Wind Speed)",
                color_continuous_scale="viridis",
            )

        elif plot_type == "3D Relationship":
            fig = px.scatter_3d(
                filtered_df,
                x="humidity",
                y="temperature",
                z="rainfall",
                color="rainfall",
                title="3D Weather Relationship",
            )

        else:  # Contour Plot
            fig = go.Figure(
                data=go.Contour(
                    x=filtered_df["humidity"],
                    y=filtered_df["temperature"],
                    z=filtered_df["rainfall"],
                    colorscale="Viridis",
                )
            )
            fig.update_layout(title="Rainfall Contour Plot (Humidity vs Temperature)")

        st.plotly_chart(fig, use_container_width=True)

        # Interactive humidity range analyzer
        st.subheader("Humidity Range Explorer")
        selected_humidity_range = st.slider(
            "Explore humidity range",
            float(filtered_df["humidity"].min()),
            float(filtered_df["humidity"].max()),
            (
                float(filtered_df["humidity"].min()),
                float(filtered_df["humidity"].max()),
            ),
        )

        # Real-time analysis for selected range
        humidity_mask = (filtered_df["humidity"] >= selected_humidity_range[0]) & (
            filtered_df["humidity"] <= selected_humidity_range[1]
        )

        with st.expander("📊 Detailed Statistics for Selected Range"):
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric(
                    "Average Rainfall",
                    f"{filtered_df[humidity_mask]['rainfall'].mean():.2f}mm",
                    f"{filtered_df[humidity_mask]['rainfall'].std():.2f}mm std",
                )
            with stats_col2:
                st.metric(
                    "Data Points",
                    len(filtered_df[humidity_mask]),
                    f"{(len(filtered_df[humidity_mask])/len(filtered_df))*100:.1f}% of total",
                )
            with stats_col3:
                st.metric(
                    "Max Rainfall",
                    f"{filtered_df[humidity_mask]['rainfall'].max():.2f}mm",
                    f"at {filtered_df[humidity_mask]['humidity'].iloc[filtered_df[humidity_mask]['rainfall'].argmax()]:.1f}% humidity",
                )

    with tab2:
        st.subheader("Dynamic Pattern Analysis")

        # Time series analysis with double y-axis
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])

        fig2.add_trace(
            go.Scatter(
                x=filtered_df["date"],
                y=filtered_df["humidity"],
                name="Humidity",
                line=dict(color="blue"),
            ),
            secondary_y=False,
        )

        fig2.add_trace(
            go.Scatter(
                x=filtered_df["date"],
                y=filtered_df["rainfall"],
                name="Rainfall",
                line=dict(color="green"),
            ),
            secondary_y=True,
        )

        fig2.update_layout(title_text="Humidity and Rainfall Time Series", height=500)

        fig2.update_xaxes(title_text="Date")
        fig2.update_yaxes(title_text="Humidity (%)", secondary_y=False)
        fig2.update_yaxes(title_text="Rainfall (mm)", secondary_y=True)

        st.plotly_chart(fig2, use_container_width=True)

        # Interactive pattern analysis
        if st.button("Generate Weather Pattern Analysis"):
            with st.spinner("Analyzing weather patterns..."):
                time.sleep(1)

                # Create humidity bins and convert to string labels
                bins = 5
                filtered_df["humidity_bin"] = pd.qcut(
                    filtered_df["humidity"],
                    q=bins,
                    labels=[f"Bin {i+1}" for i in range(bins)],
                )

                # Calculate statistics
                rainfall_stats = (
                    filtered_df.groupby("humidity_bin")
                    .agg({"rainfall": ["mean", "count", "std"]})
                    .round(2)
                )

                rainfall_stats.columns = ["mean", "count", "std"]
                rainfall_stats = rainfall_stats.reset_index()

                # Create the bar plot
                fig3 = px.bar(
                    rainfall_stats,
                    x="humidity_bin",
                    y="mean",
                    error_y="std",
                    title="Average Rainfall by Humidity Ranges",
                    labels={
                        "humidity_bin": "Humidity Range",
                        "mean": "Average Rainfall (mm)",
                    },
                )

                fig3.update_layout(
                    xaxis_title="Humidity Range",
                    yaxis_title="Average Rainfall (mm)",
                    showlegend=True,
                )

                st.plotly_chart(fig3, use_container_width=True)

                # Display additional statistics
                st.subheader("Detailed Statistics by Humidity Range")

                # Create a more readable format for the stats
                stats_df = pd.DataFrame(
                    {
                        "Humidity Range": rainfall_stats["humidity_bin"],
                        "Average Rainfall (mm)": rainfall_stats["mean"],
                        "Number of Observations": rainfall_stats["count"],
                        "Standard Deviation": rainfall_stats["std"],
                    }
                )

                st.dataframe(stats_df, use_container_width=True)

                # Add insights based on the analysis
                st.subheader("Key Findings")

                # Find the range with maximum rainfall
                max_rainfall_bin = stats_df.loc[
                    stats_df["Average Rainfall (mm)"].idxmax()
                ]

                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    Highest Average Rainfall:
                    - Range: {max_rainfall_bin['Humidity Range']}
                    - Amount: {max_rainfall_bin['Average Rainfall (mm)']:.2f} mm
                    """)

                with col2:
                    # Calculate overall trend
                    first_bin = stats_df["Average Rainfall (mm)"].iloc[0]
                    last_bin = stats_df["Average Rainfall (mm)"].iloc[-1]
                    trend = "increasing" if last_bin > first_bin else "decreasing"

                    st.info(f"""
                    Overall Trend:
                    - Pattern: {trend.title()}
                    - Range: {first_bin:.2f}mm to {last_bin:.2f}mm
                    """)

                # Add correlation analysis
                correlation = filtered_df["rainfall"].corr(filtered_df["humidity"])
                st.metric(
                    label="Humidity-Rainfall Correlation",
                    value=f"{correlation:.2f}",
                    delta="Strong positive correlation"
                    if correlation > 0.5
                    else "Moderate correlation",
                )

                # Optional: Add distribution plot
                if st.checkbox("Show Distribution Plot"):
                    fig4 = px.histogram(
                        filtered_df,
                        x="humidity",
                        nbins=20,
                        title="Humidity Distribution",
                        color_discrete_sequence=["blue"],
                    )
                    st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.subheader("Real-time Humidity Insights")

        # Interactive humidity threshold analysis
        threshold_humidity = st.slider(
            "Humidity Threshold (%)",
            float(filtered_df["humidity"].min()),
            float(filtered_df["humidity"].max()),
            float(filtered_df["humidity"].mean()),
        )

        # Animated analysis results
        with st.spinner("Calculating threshold statistics..."):
            time.sleep(0.5)
            above_threshold = filtered_df[filtered_df["humidity"] > threshold_humidity][
                "rainfall"
            ].mean()
            below_threshold = filtered_df[
                filtered_df["humidity"] <= threshold_humidity
            ]["rainfall"].mean()

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    f"Avg Rainfall Above {threshold_humidity:.1f}%",
                    f"{above_threshold:.2f}mm",
                    f"{above_threshold - below_threshold:.2f}mm difference",
                )
            with col2:
                st.metric(
                    f"Avg Rainfall Below {threshold_humidity:.1f}%",
                    f"{below_threshold:.2f}mm",
                    f"{below_threshold - above_threshold:.2f}mm difference",
                )

        # Dynamic insights generation
        st.subheader("Key Weather Insights")
        insights = [
            f"Peak rainfall occurs at {filtered_df.loc[filtered_df['rainfall'].idxmax(), 'humidity']:.1f}% humidity",
            f"Humidity range {selected_humidity_range[0]:.1f}% to {selected_humidity_range[1]:.1f}% shows {filtered_df[humidity_mask]['rainfall'].mean():.1f}mm average rainfall",
            f"Analysis based on {len(filtered_df)} weather observations",
        ]

        for i, insight in enumerate(insights):
            with st.spinner(f"Loading insight {i+1}/3..."):
                time.sleep(0.3)
                st.info(insight)

        # Seasonal pattern analysis
        if st.checkbox("Show Seasonal Patterns"):
            filtered_df["month"] = filtered_df["date"].dt.month
            seasonal_data = filtered_df.groupby("month")[
                ["humidity", "rainfall"]
            ].mean()

            fig4 = px.line(
                seasonal_data,
                title="Monthly Humidity and Rainfall Patterns",
                labels={"value": "Average Value", "month": "Month"},
                color_discrete_sequence=["blue", "green"],
            )
            st.plotly_chart(fig4, use_container_width=True)

        # Add download button with animation
        if st.download_button(
            label="Download Complete Analysis Report",
            data=filtered_df.to_csv().encode("utf-8"),
            file_name="humidity_rainfall_analysis.csv",
            mime="text/csv",
        ):
            st.balloons()
            st.success("Analysis report downloaded successfully!")


def show_overall_insights(df):
    st.subheader("Key Weather Pattern Insights")

    st.markdown("""
    ### Temperature Impact
    - Inverse relationship with rainfall
    - Critical temperature thresholds affect rainfall probability
    - Higher temperatures generally mean lower rainfall chances

    ### Humidity Impact
    - Direct relationship with rainfall
    - Strong predictor of rainfall probability
    - Clear threshold points for rainfall occurrence

    ### Combined Effects
    Different combinations of temperature and humidity create distinct weather patterns:
    - Low temperature + high humidity = Highest rainfall probability
    - High temperature + low humidity = Lowest rainfall probability
    - Moderate values = Variable rainfall patterns
    """)


if __name__ == "__main__":
    main()

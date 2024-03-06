import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from PIL import Image

# Loading the dataset
def load_data():
    df = pd.read_csv('spotify.csv', encoding='latin-1')

    # Clean the 'streams' column
    df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
    mean_value = round(df['streams'].mean())

    # Replace NaN values with the rounded mean
    df['streams'] = df['streams'].fillna(mean_value)

    # Convert the column to integer type
    df['streams'] = df['streams'].astype(int)

    cleaned_data = df.dropna(subset=['key', 'in_shazam_charts'])

    return cleaned_data

df = load_data()

# Set the title
st.title("Welcome to Spotify's predictive dashboard!")


# load spotify image
image = Image.open('spotify.png')

# display image
st.image(image, caption='Spotify Image', use_column_width=True)



# Sidebar
st.sidebar.header('Select Page')

# Sidebar - Page selection
page = st.sidebar.selectbox('Select', ['Background', 'Analysis', 'Predictions'])

# display selected page
if page == 'Background':
    # displaying background info
    st.subheader('Objectives')
    st.write("🎯 Spotify is one of the largest music streaming service providers, with over 602 million monthly active users, including 236 million paying subscribers, as of December 2023. The goal of this project is to allow different businesses such as record labels to make data-driven decisions based on the dataset, highlighting key characteristics of hit songs in 2023.")

    st.subheader('Dataset')
    st.write("🎯 What Spotify has available is it's most streamed songs in 2023. It provides insights into each song's attributes, popularity, and presence on various music platforms.")
    st.write(df.head())
    st.write("Source: https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023")

elif page == 'Analysis':
    columns_to_heatmap = ['in_spotify_playlists', 'streams', 'bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']

    # filter the DataFrame to include only the specified columns
    filtered_df = df[columns_to_heatmap]

    # calculate the correlation matrix for the filtered DataFrame
    corr = filtered_df.corr()

    #1st pairplot image path
    pairplot_image_path = 'pairplot1.png'

    # display the image
    st.image(pairplot_image_path, caption='Pairplot 1')

    # create the heatmap
    fig = px.imshow(
        corr,
        text_auto=True,
        labels=dict(x="Feature", y="Feature", color="Correlation"),
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale='Viridis',
    )

    fig.update_layout(
        title_text='Heatmap of Feature Correlations',
        title_x=0  # title aligned to the left
    )

    # display
    st.plotly_chart(fig)

    # pairplot image path
    pairplot_image_path = 'pairplot.png'

    # display the image
    st.image(pairplot_image_path, caption='Pairplot 2')

    # tabs for each visualization
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Count of Each Key", "Average Streams by Key", "BPM Histogram", "Artist Chart", "Number of Artists Contributing to Each Song"])

    with tab1:
        st.subheader("Count of Each Key")

        # calculate counts for each key
        key_counts = df['key'].value_counts().reset_index()
        key_counts.columns = ['key', 'count']

        # interactive bar chart
        fig = px.bar(
            key_counts,
            x='key',
            y='count',
            color='key',
            labels={'count': 'Count', 'key': 'Key'},
            title="Count of Each Key",
            hover_data={'count': True},
        )

        # layout
        fig.update_layout(
            showlegend=False,
            coloraxis_colorbar=dict(title='Key'),
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # display
        st.plotly_chart(fig)

        st.write("👨‍💻 Here we can see that the most prevalent key among popular songs is C#, which tells us that this is a very popular key for hit songs in 2023.")

    with tab2:
        st.subheader('Average Streams by Key')
        new_df = df.groupby('key')['streams'].agg(['mean', 'min', 'max']).reset_index()
        new_df = new_df.rename(columns={'mean': 'avg_streams', 'min': 'min_streams', 'max': 'max_streams'})

        fig = px.bar(
            new_df,
            x='key',
            y='avg_streams',
            color='key',
            labels={'avg_streams': 'Average Streams', 'key': 'Key'},
            title='Average Streams by Key',
            hover_data={'avg_streams': True, 'min_streams': False, 'max_streams': False},
        )

        # layout
        fig.update_layout(
            xaxis={'categoryorder': 'total descending'},
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
        )

        # Display the figure
        st.plotly_chart(fig)

        st.write("👨‍💻 Here we can see that the average streams by key are highest with C#, though the second is E. That tells us that even though E is not the most prevalent in the dataset, the average of the songs in the key of E have a lot of streams. This graph tells us something different than the Count of Each Key - this way we can see which song keys are streamed the most.")

    with tab3:
        st.subheader('BPM Histogram')
        fig = px.histogram(df, x='bpm', nbins=50, title="Distribution of BPM")


        fig.update_layout(
            xaxis_title="BPM",
            yaxis_title="Count",
            bargap=0.2,
        )

        # display
        st.plotly_chart(fig)

        st.write("👨‍💻 Here we can see that the most popular BPM's are in the 120-124 range.")
    with tab4:
        st.subheader('Artist Chart')
        # splitting artist names and putting them into separate rows
        all_artists = df['artist(s)_name'].str.split(', ').explode()

        # counting occurrences of each artist
        artist_counts = all_artists.value_counts().reset_index()
        artist_counts.columns = ['Artist', 'Frequency']

        # visualizing top 10 artists
        fig = px.bar(artist_counts.head(10), x='Artist', y='Frequency',
                    title='Top 10 Artists by Frequency on Chart',
                    labels={'Frequency': 'Number of Appearances'},
                    color='Frequency',
                    color_continuous_scale=px.colors.sequential.Viridis)

        # display
        st.plotly_chart(fig)
        st.write("👨‍💻 Here we can see the most popular artists of 2023.")
    with tab5:
        st.subheader('Number of Artists Contributing to Each Song')
        artist_count_distribution = df['artist_count'].value_counts().reset_index()
        artist_count_distribution.columns = ['Number of Artists', 'Number of Songs']
        artist_count_distribution = artist_count_distribution.sort_values('Number of Artists')
        # visualizing the distribution of the number of artists contributing to each song
        fig = px.bar(artist_count_distribution, x='Number of Artists', y='Number of Songs',
                    title='Number of Artists Contributing to Each Song',
                    labels={'Number of Songs': 'Number of Songs', 'Number of Artists': 'Number of Artists'},
                    color='Number of Songs',
                    color_continuous_scale=px.colors.sequential.Viridis)

        # display
        st.plotly_chart(fig)
        st.write("👨‍💻 Here we can see how different artists contribute to the most popular songs of 2023.")
elif page == 'Predictions':
    df['danceability_energy'] = df['danceability_%'] * df['energy_%']
    df['valence_danceability'] = df['valence_%'] / df['danceability_%']

    # select only numeric features for modeling
    quantitative_df = df.select_dtypes(include=[np.number])

    # preparing the data
    X = quantitative_df.drop('streams', axis=1)
    y = quantitative_df['streams']

    # splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # training the linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # displaying feature importance
    coefficients = lin_reg.coef_
    feature_names = X.columns
    importance = np.abs(coefficients)

    # plotting feature importance
    fig_importance = px.bar(x=importance, y=feature_names, orientation='h',
                            labels={'x': 'Absolute Coefficient Value', 'y': ''},
                            title='Feature Importance (Linear Regression)')
    st.plotly_chart(fig_importance)

    # making predictions
    pred = lin_reg.predict(X_test)

    # plotting actual vs predicted streams
    fig_pred = px.scatter(x=y_test, y=pred, labels={'x': 'Actual Streams', 'y': 'Predicted Streams'},
                          title="Actual vs. Predicted Streams", trendline="ols")
    fig_pred.add_shape(type='line', line=dict(dash='dash', color='red'),
                      x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
    st.plotly_chart(fig_pred)

    # displaying metrics
    mae = metrics.mean_absolute_error(y_test, pred)
    mse = metrics.mean_squared_error(y_test, pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
    r2 = metrics.r2_score(y_test, pred)

    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    st.write(f"R^2 Score: {r2}")

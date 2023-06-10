import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from wordcloud import WordCloud
import plotly.graph_objects as go


 # Import dataset to Data Frame
data = pd.read_csv("./Data/movie_metadata.csv")

col1, col2 = st.columns(2)
#1. Top 10 genre
top_genres = data['genres'].value_counts().iloc[:10].index.tolist()
genre_counts = data['genres'].value_counts().iloc[:10].values.tolist()

# sort the data in descending order
sorted_indices = sorted(range(len(genre_counts)), key=lambda k: genre_counts[k], reverse=False)
sorted_genres = [top_genres[i] for i in sorted_indices]
sorted_counts = [genre_counts[i] for i in sorted_indices]

top_ten_movies = px.bar(
    x=sorted_counts, 
    y=sorted_genres, 
    orientation='h', 
    title='Top 10 Movie Genres', 
    labels={'x': 'Number of Movies', 'y': 'Genres'},
    color=sorted_counts,
    color_continuous_scale='magma'
)

with col1:
    st.plotly_chart(top_ten_movies)


# 2. Get the top 10 countries by movie count
top_countries = data['country'].value_counts().iloc[:10].index.tolist()
country_counts = data['country'].value_counts().iloc[:10].values.tolist()

# Create a custom color palette
colors = px.colors.qualitative.Pastel

# Create the bar plot using Plotly
top_10_countries = px.bar(
    x=top_countries,
    y=country_counts,
    color=top_countries,
    color_discrete_sequence=colors,
    title="Top 10 Countries by Movie Count",
    labels={'x': 'Country', 'y': 'Movie Count'},
)
top_10_countries.update_layout(
    xaxis_tickangle=-45,
    xaxis_tickfont=dict(size=12),
    yaxis_tickfont=dict(size=12),
    yaxis_range=[0, max(country_counts) + 10],
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)'
)
with col2:
    st.plotly_chart(top_10_countries)

col3, col4 = st.columns(2)
# 3. top 10 release years
# Fill missing values with 0 and convert to int64
data['title_year'].fillna(0, inplace=True)
data['title_year'] = data['title_year'].astype(np.int64)

# Get the top 10 release years
top_years = data['title_year'].value_counts().head(10)

# Create the bar plot using Plotly
top_10_release_years = px.bar(
    x=top_years.index,
    y=top_years.values,
    color=top_years.index,
    color_continuous_scale='Blues',
    title='Top 10 Movie Release Years',
    labels={'x': 'Year', 'y': 'Number of Movies'},
)

top_10_release_years.update_layout(
    xaxis_tickfont=dict(size=12),
    yaxis_tickfont=dict(size=12),
    yaxis_range=[0, max(top_years.values) + 10],
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)'
)
with col3:
    st.plotly_chart(top_10_release_years)


# 4. Genre_Distribution
# Create a new column with the genre name only
data['genre'] = data['genres'].apply(lambda x: x.split('|')[0])

# Calculate the frequency of each genre and create a DataFrame
genre_counts = data['genre'].value_counts().reset_index()
genre_counts.columns = ['genre', 'count']

# Create the treemap using Plotly
Genre_Distribution = px.treemap(
    data_frame=genre_counts,
    path=['genre'],
    values='count',
    color='count',
    color_continuous_scale='Blues',
    title='Genre Distribution',
)

Genre_Distribution.update_layout(
    margin=dict(l=10, r=10, t=30, b=10),
    plot_bgcolor='white'
)
with col4:
    st.plotly_chart(Genre_Distribution)


# col3, col4 = st.columns(2)
# 5. wordcloud
# plot unique cast frequency
language_info = data["language"].value_counts()[:1000]

myWordcloud = WordCloud(background_color='black')
myWordcloud.generate_from_frequencies(dict(language_info))

myWordcloud = go.Figure(go.Image(z=myWordcloud))
myWordcloud.update_layout(title="Origin Language", width=800, height=700)
myWordcloud.update_xaxes(visible=False)
myWordcloud.update_yaxes(visible=False)
st.plotly_chart(myWordcloud)

#Model Performance
# Define the labels and data for the bar chart
labels = ['Precision', 'Recall', 'F1-score', 'Accuracy']
m_data = [0.75, 0.72, 0.69, 0.72]

# Create a bar trace with a viridis color scale
bar_trace = go.Bar(x=labels, y=m_data, marker=dict(color=m_data, colorscale='viridis'))

# Create a layout and set the chart title and axis labels
layout = go.Layout(title='Model Performance', xaxis=dict(title='Metrics'), yaxis=dict(title='Scores'))

# Create the figure and add the bar trace and layout
model_perf = go.Figure(data=[bar_trace], layout=layout)
st.plotly_chart(model_perf)






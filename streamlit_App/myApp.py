from lib2to3.pytree import convert
import streamlit as st
from PIL import Image
import json
from Classifier import KNearestNeighbours
from bs4 import BeautifulSoup
import requests,io
import PIL.Image
from urllib.request import urlopen
import pandas as pd
import base64

with open('./Data/movie_data.json', 'r+', encoding='utf-8') as f:
    data = json.load(f)
with open('./Data/movie_titles.json', 'r+', encoding='utf-8') as f:
    movie_titles = json.load(f)

header = {"User-Agent" : 
           "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"}


def movie_poster_fetcher(imdb_link):
    ## Display Movie Poster
    url_data = requests.get(imdb_link, headers = header).text
    s_data = BeautifulSoup(url_data, 'html.parser')
    # print(s_data)

    imdb_dp = s_data.find("meta", property="og:image")
    movie_poster_link = imdb_dp.attrs['content']
    u = urlopen(movie_poster_link)
    raw_data = u.read()
    image = PIL.Image.open(io.BytesIO(raw_data))
    # image = image.resize((158, 301), )
    image = image.resize((400, 500), )
    return image
   

def get_movie_info(imdb_link):
    url_data = requests.get(imdb_link, headers = header).text
    s_data = BeautifulSoup(url_data, 'html.parser')
    imdb_content = s_data.find("meta", property="og:description")
    movie_descr = imdb_content.attrs['content']
    movie_descr = str(movie_descr).split('.')
    movie_director = movie_descr[0]
    movie_cast = str(movie_descr[1]).replace('With', 'Cast: ').strip()
    movie_story = 'Story: ' + str(movie_descr[2]).strip()+'.'
    

    # #ID at the top of the total rating id. It brings n value in {Just commented this a week before defence close to habeeb}
    # rating = s_data.find("div", class_="sc-e457ee34-5 gxhMxc")
    # rating = str(rating).split('<div class="sc-e457ee34-5 gxhMxc')

     #ID at the top of the total rating id. It brings n value in {Just Added this to check it out}
    rating = s_data.find("div", class_="sc-bde20123-3 bjjENQ")
    rating = str(rating).split('<div class="sc-bde20123-3 bjjENQ"')
    

    
    rating = str(rating[1]).split("</div>")
    rating = str(rating[0]).replace(''' "> ''', '').replace('">', '')

    movie_rating = 'Total Rating count: '+ rating
    return movie_director,movie_cast,movie_story,movie_rating

def KNN_Movie_Recommender(test_point, k):
    # Create dummy target variable for the KNN Classifier
    target = [0 for item in movie_titles]
    # Instantiate object for the Classifier
    model = KNearestNeighbours(data, target, test_point, k=k)
    # Run the algorithm
    model.fit()
    # Print list of 10 recommendations < Change value of k for a different number >
    table = []
    for i in model.indices:
        # Returns back movie title and imdb link
        table.append([movie_titles[i][0], movie_titles[i][2],data[i][-1]])
    print(table)
    return table

st.set_page_config(
   page_title="Film Forecast",)
# st.sidebar.success("Select a Page")

raw_data_df = pd.read_csv("./Data/movie_metadata.csv")
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(raw_data_df)
def run():
    # img1 = Image.open('./meta/FF_Logo.png')
    # img1 = img1.resize((300,80),)
    # colu1, colu2, colu3, colu4 = st.columns(4)
    # with colu2:
    #     st.image(img1,use_column_width=False)
    # Download dataset
    st.markdown('''<h5 style='text-align: left; color: #ffbf3a;'>The Recommendation Engine was built based on</h5>''',
                    unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Movies", "5K+", "")
    col2.metric("Genres", "26", "")
    col3.metric("Year", "2023", "")
    with col4:
        st.download_button(
            label="Download Data",
            data=csv,
            file_name='Movie Raw Data.csv',
            mime='text/csv',
            )
      

    # st.title("Movie Recommender System")
    # st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "IMDB 5000 Movie Dataset"</h4>''',
                # unsafe_allow_html=True)
    # df = pd.read_csv("C:\Users\DELL\Documents\FINAL_FINAL FINAL_PROJECT_APPLICATION\Data\movie_metadata.csv")
    # st.dataframe(df)
    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
              'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
              'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    movies = [title[0] for title in movie_titles]
    category = ['--Select--', 'Movie based', 'Genre based']
    cat_op = st.selectbox('Select Recommendation Type', category)
    if cat_op == category[0]:
        st.warning('Please select Recommendation Type!!')
    elif cat_op == category[1]:
        select_movie = st.selectbox('Select movie: (Recommendation will be based on this selection)', ['--Select--'] + movies)
        dec = st.radio("Want to Fetch Movie Poster?", ('Yes', 'No'))
        st.markdown('''<h5 style='text-align: left; color: #ffbf3a;'>Fetching Movie Posters take few seconds...</h5>''',
                    unsafe_allow_html=True)
        if dec == 'No':
            if select_movie == '--Select--':
                st.warning('Please select Movie!!')
            else:
                no_of_reco = st.slider('Number of movies you want Recommended:', min_value=5, max_value=20, step=1)
                genres = data[movies.index(select_movie)]
                test_points = genres
                table = KNN_Movie_Recommender(test_points, no_of_reco+1)
                table.pop(0)
                c = 0
                st.success('Some of the movies from our Recommendation, have a look below')
                for movie, link, ratings in table:
                    c+=1
                    director,cast,story,total_rat = get_movie_info(link)
                    # with col2:
                    st.markdown(f"({c})[ {movie}]({link})")
                    st.markdown(director)
                    st.markdown(cast)
                    st.markdown(story)
                    st.markdown(total_rat)
                    st.markdown('IMDB Rating: ' + str(ratings) + '⭐')
                    st.markdown("#")
        else:
            
            if select_movie == '--Select--':
                st.warning('Please select Movie!!')
            else:
                no_of_reco = st.slider('Number of movies you want Recommended:', min_value=5, max_value=20, step=1)
                genres = data[movies.index(select_movie)]
                test_points = genres
                table = KNN_Movie_Recommender(test_points, no_of_reco+1)
                table.pop(0)
                c = 0
                st.success('Some of the movies from our Recommendation, have a look below')
                for movie, link, ratings in table:
                    c += 1
                    # st.markdown(f"({c})[ {movie}]({link})")
                    col1, col2 =  st.columns([1,2])
                    img = movie_poster_fetcher(link)
                    with col1:
                        st.image(img, use_column_width=True)
                    director,cast,story,total_rat = get_movie_info(link)
                    with col2:
                        st.markdown(f"({c})[ {movie}]({link})")
                        st.markdown(director)
                        st.markdown(cast)
                        st.markdown(story)
                        st.markdown(total_rat)
                        st.markdown('IMDB Rating: ' + str(ratings) + '⭐')
    elif cat_op == category[2]:
        sel_gen = st.multiselect('Select Genres:', genres)
        dec = st.radio("Want to Fetch Movie Poster?", ('Yes', 'No'))
        st.markdown('''<h5 style='text-align: left; color: #ffbf3a;'>*Fetching Movie Posters take few seconds...</h5>''',
                    unsafe_allow_html=True)
        if dec == 'No':
            if sel_gen:
                imdb_score = st.slider('Choose IMDb score:', 1, 10, 8)
                no_of_reco = st.number_input('Number of movies:', min_value=5, max_value=20, step=1)
                test_point = [1 if genre in sel_gen else 0 for genre in genres]
                test_point.append(imdb_score)
                table = KNN_Movie_Recommender(test_point, no_of_reco)
                c = 0
                st.success('Some of the movies from our Recommendation, have a look below')
                for movie, link, ratings in table:
                    c += 1
                    st.markdown(f"({c})[ {movie}]({link})")
                    director,cast,story,total_rat = get_movie_info(link)
                    st.markdown(director)
                    st.markdown(cast)
                    st.markdown(story)
                    st.markdown(total_rat)
                    st.markdown('IMDB Rating: ' + str(ratings) + '⭐')
        else:
            if sel_gen:
                imdb_score = st.slider('Choose IMDb score:', 1, 10, 8)
                no_of_reco = st.number_input('Number of movies:', min_value=5, max_value=20, step=1)
                test_point = [1 if genre in sel_gen else 0 for genre in genres]
                test_point.append(imdb_score)
                table = KNN_Movie_Recommender(test_point, no_of_reco)
                c = 0
                st.success('Some of the movies from our Recommendation, have a look below')
                for movie, link, ratings in table:
                    c += 1
                    # st.markdown(f"({c})[ {movie}]({link})")
                    col1, col2 =  st.columns([1,2])
                    img = movie_poster_fetcher(link)
                    with col1:
                        st.image(img, use_column_width=True)
                    director,cast,story,total_rat = get_movie_info(link)
                    with col2:
                        st.markdown(f"({c})[ {movie}]({link})")
                        st.markdown(director)
                        st.markdown(cast)
                        st.markdown(story)
                        st.markdown(total_rat)
                        st.markdown('IMDB Rating: ' + str(ratings) + '⭐')
                        st.markdown("#")
run()
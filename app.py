import streamlit as st
import pandas as pd
import numpy as np
import operator
from scipy import spatial
import json
from PIL import Image
import requests
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from st_files_connection import FilesConnection

conn = st.connection('s3', type=FilesConnection)
df = conn.read("streamlitbucketjm/filtered_games.csv", input_format="csv", ttl=600)
df_2 = conn.read("streamlitbucketjm/filtered_data.json", input_format="json")
#with open(df_2, 'r') as json_file:
dict_dataset = df_2
df_col = pd.DataFrame(df_2)

#df=pickle.load(open('movie_list.pkl','rb'))
# Let's open the file and load the data
#df = pd.read_csv('./games.csv')
#with open('./data_recommendation.json', 'r') as json_file:
#    dict_dataset = json.load(json_file)
#    df_col = pd.DataFrame(dict_dataset)

st.title("Game Recommendation System")


def Computedistnace_themes(a, b):
    themesDistance = spatial.distance.cosine(a['themes'], b['themes'])
    popularityDistance = abs(a['NumUserRatings'] - b['NumUserRatings'])
    genreDistance = spatial.distance.cosine(a['Categories'], b['Categories'])
    clusterDistance = spatial.distance.cosine(a['l_cluster'], b['l_cluster'])
    descDistance = spatial.distance.cosine(a['emb'], b['emb'])
    ratingDistance = abs(a['BayesAvgRating'] - b['BayesAvgRating'])
    return descDistance + ratingDistance + popularityDistance*0.9 + clusterDistance*0.03 + themesDistance*0.04

def getNeighbors_themes(baseGame, k):
    distances = []
    for game in dict_dataset:
        if game['BGGId'] != baseGame['BGGId']:
            dist = Computedistnace_themes(baseGame, game)
            distances.append((game, dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = [x[0] for x in distances[:k]]
    return neighbors

def get_recommendations(games_list, k_themes=20, top=5):
    recommendations_dict = {}

    for game in games_list:
        neighbors = getNeighbors_themes(game, k_themes)
        avgRating_themes = np.mean([neighbor['BayesAvgRating'] for neighbor in neighbors])
        recommendations = [neighbor['Name'] for neighbor in neighbors[:top]]
        recommendations_dict[game['Name']] = recommendations

    return recommendations_dict

# Let's create a transformation function that takes the input of the categories and returns the the one-hot encoded list
def transform_categories(categories):
    """
    Function to transform the categories into a one-hot encoded list to use it in the search engine
    """
    # Create a list with all the categories
    all_categories = ['thematic', 'strategy', 'war', 'family', 'cgs', 'abstract', 'party', 'childrens']
    for i in range(len(all_categories)):
        if all_categories[i] in categories:
            all_categories[i] = 1
        else:
            all_categories[i] = 0

    # Let's get the index where 1 is in the list a
    index = [i for i in range(len(all_categories)) if all_categories[i] == 1]
    return index

def index_of_data(index, categories):
    """
    Function to return the index of the list that contains the index
    """
    index_list = []
    for i in range(len(categories)):
        if any(categories[i][j] == 1 for j in index):
            # let's print the index of the list
            index_list.append(i)

    return index_list

def get_new_dataset(user_input, df):
    """
    Function to return the new dataset filter with the index list
    """
    index = transform_categories(user_input)
    index_list = index_of_data(index, df['Categories'])
    dataset = df.loc[index_list]
    return dataset

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_cluster(user_input, new_dataset_col):
    """
    """
    to_filter = get_new_dataset(user_input, new_dataset_col)
    to_filter[['x1', 'x2']] = to_filter['emb'].apply(lambda x: pd.Series(x, index=['x1', 'x2']))

    # Selecting the features for clustering
    features_for_clustering = ['x1', 'x2']

    # Extracting feature values from the dataframe
    feature_values = to_filter[features_for_clustering].apply(np.hstack, axis=1).tolist()

    feature_values = np.array(feature_values)

    # Determine the optimal number of clusters using the Elbow Method
    silhouette_scores = []
    possible_clusters = range(2, 10)  # You can adjust the range as needed

    for num_clusters in possible_clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(feature_values)
        silhouette_avg = silhouette_score(feature_values, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Find the optimal number of clusters based on the Silhouette Method
    optimal_num_clusters = np.argmax(silhouette_scores) + 2

    # Number of clusters (you can adjust this based on your requirements)
    num_clusters = optimal_num_clusters

    # K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    to_filter['cluster'] = kmeans.fit_predict(feature_values)

    # Now that I get the dataset with the clusters, let's sort the dataset by the BayesAvgRating
    first_half = to_filter.loc[to_filter['cluster'] == 0].sort_values(by='BayesAvgRating', ascending=False).head(5)
    second_half = to_filter.loc[to_filter['cluster'] == 1].sort_values(by='BayesAvgRating', ascending=False).head(5)

    # Let's get the BGGId of the first and second half
    first_half_id = first_half['BGGId'].values
    second_half_id = second_half['BGGId'].values

    return first_half_id, second_half_id

with st.sidebar:
    bggid_url = "./logo-no-background.png"
    st.image(bggid_url, width=250)
    
recommendation_type = 'By game'  # Set the default recommendation type

# If the user selects 'By game'
if recommendation_type == 'By game':
    selected_games = st.multiselect(
        'Type or select three games from the dropdown',
        list(df.Name.values),
        default=None
    )

    # Let's get the index of the selected movies
    selected_games_index = []
    for i in range(len(selected_games)):
        index_ = df[df.Name == selected_games[i]].index[0]
        selected_games_index.append(dict_dataset[index_])

    if selected_games == None or len(selected_games_index) == 0:
        st.warning("Please select games")
    elif st.button("Get recommendation"):
        #st.markdown("The selected games are: " + ", ".join(selected_games))

        dictio = get_recommendations(selected_games_index, top=10)

        for key in dictio:
            with st.container():
                st.text(f"Because you chose: {key}")
                
                # Create two rows of 5 columns
                row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns([5, 5, 5, 5, 5])
                row2_col1, row2_col2, row2_col3, row2_col4, row2_col5 = st.columns([5, 5, 5, 5, 5])

                for idx, name in enumerate(dictio[key]):
                    if idx < 5:
                        current_col = row1_col1 if idx == 0 else row1_col2 if idx == 1 else row1_col3 if idx == 2 else row1_col4 if idx == 3 else row1_col5
                    else:
                        current_col = row2_col1 if idx == 5 else row2_col2 if idx == 6 else row2_col3 if idx == 7 else row2_col4 if idx == 8 else row2_col5
                    
                    with current_col:
                        text = dictio[key][idx]
                        bottom_image_url = requests.get((df[df.Name == dictio[key][idx]].iloc[:, -18].values[0]))
                        if bottom_image_url is not None:
                            image = Image.open(BytesIO(bottom_image_url.content))
                            new_image = image.resize((300, 200))
                            st.image(new_image, caption=text)
                st.divider()
                
else:
    st.warning("Please select a recommendation type")
    st.stop()

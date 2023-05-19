# Imports
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from SpotifyRecommender import SpotifyRecommender
import difflib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# SPOTIFY API
client_id = "7dd1627b4bcd44d494ca70d954e02eaf"
client_secret = "55fa312e777e49669556c038ab11906e"

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# SEARCH + DISPLAY
# Get a list of all available genres
genre_list = sp.recommendation_genre_seeds()['genres']

# Add a text input for the search query
query = st.text_input("Search for a track")

# Add a select box for the genres
selected_genre = st.selectbox("Select a genre to filter by", options=genre_list)

# Add a toggle button for the genre filter
filter_on = st.checkbox('Turn on genre filter')

# If a query is entered, perform the search
if query:
    # Add the selected genre to the query if the filter is turned on
    if filter_on and selected_genre:
        query += f" genre:{selected_genre}"

    # Perform the search
    results = sp.search(query, type='track', limit=10)

    # Create a list of track ids
    track_ids = [track['id'] for track in results['tracks']['items']]

    # Display the tracks and store the selected track info
    track_infos = [f"{track['name']} by {track['artists'][0]['name']}" for track in results['tracks']['items']]
    radio_options = {track_info: i for i, track_info in enumerate(track_infos)}
    selected_track_info = st.radio("Select a track", options=track_infos)

    # Get the track_id of the selected track
    if selected_track_info is not None:
        selected_track_id = track_ids[radio_options[selected_track_info]]
        # You can use the selected_track_id variable as needed
        st.write(f"Selected track ID: {selected_track_id}")


# FEATURES FROM SPOTIFY
"""
test_track_id = "7KXjTSCq5nL1LoYtL7XAwS"
features = sp.audio_features(test_track_id)
analysis = sp.audio_analysis(test_track_id)

# Save in the same dictionary
track_data = {
    "features": features[0] if features else None,
    "analysis": analysis
}

import json

with open('track_data.json', 'w') as json_file:
    json.dump(track_data, json_file)
"""

# MODEL CALL


# LIST RECOMMENDED TRACKS
# Will be filled by model
track_ids = ['5qljLQuKnNJf4F4vfxQB0V', '3VAX2MJdmdqARLSU5hPMpm', '1L3YAhsEMrGVvCgDXj2TYn']

if track_ids:  # This checks if the track_ids list is not empty
    for track_id in track_ids:
        # Get the Spotify track details
        track = sp.track(track_id)

        # Extract the image URL, preview URL, track name and artist name from the track details
        image_url = track['album']['images'][0]['url']
        preview_url = track['preview_url']
        track_name = track['name']
        artist_name = track['artists'][0]['name']

        # Create a container with a box around it
        with st.container():
            # Create three columns
            col1, col2, col3 = st.columns(3)

            # Display the image in the first column
            col1.image(image_url, width=80)

            # Display the track name and artist name in the second column
            col2.write(f"{track_name}\n by {artist_name}")

            # If a preview URL is available, display the audio player in the third column
            if preview_url is not None:
                col3.audio(preview_url)
            else:
                col3.write("No preview available")


"""
# Load the model from pickle
model = SpotifyRecommander(dataset=model_data)
model.load('./tests/2023-05-14_model_allfeatures_david.pickle')

# Get recommendations
if selected_song:
    track_id = model_data[model_data['song_name'] == selected_song]['track_id'].values[0]
    recommendations = model.predict(track_id)
    st.write("Recommended songs:")
    st.write(recommendations)


# Filter recommendations by genre
if not recommendations.empty:
    genres = recommendations['genre'].unique()
    selected_genre = st.selectbox('Filter by genre:', ['All'] + list(genres))
    if selected_genre != 'All':
        filtered_recommendations = recommendations[recommendations['genre'] == selected_genre]
        st.write("Filtered recommendations:")
        st.write(filtered_recommendations)


# Display list of all tracks if all genres
if st.button("Show all tracks"):
    st.write("All tracks:")
    st.write(data)

# OLD
# SEARCH TRACKS
def get_similar_results(df, search_str):
    # Compute similarity scores using difflib for each row
    df['similarity_score'] = df.apply(lambda row: max(
        difflib.SequenceMatcher(None, search_str.lower(), row['album_name']).ratio(),
        difflib.SequenceMatcher(None, search_str.lower(), row['artist_name']).ratio(),
        difflib.SequenceMatcher(None, search_str.lower(), row['track_name']).ratio()
    ), axis=1)

    # Sort the dataframe by similarity score in descending order
    sorted_df = df.sort_values(by='similarity_score', ascending=False)

    # Return the track_id of the top 10 most similar rows
    return sorted_df['track_id'].head(10).tolist()


def get_rows_by_track_ids(df, track_ids):
    return df[df['id'].isin(track_ids)]


def search_container(input_df, output_df, search_track):
    results = get_similar_results(input_df, search_track)
    return get_rows_by_track_ids(output_df, results)


# Streamlit interface with search box and list of songs
st.title("Spotify Song Recommender")

# Search for songs
search_query = st.text_input("Search for a song:")
if search_query:
    search_results = search_container(my_input_df, my_output_df, search_query)
    if not search_results.empty:
        st.write("Search results:")

        selected_song = None
        for index, row in search_results.iterrows():
            song_description = f"{row['name']} - {row['id']}"
            if st.button(song_description):
                selected_song = row['album_id']
                break

        if selected_song is not None:
            st.write(f"Selected song: {selected_song}")
        else:
            st.write("No song selected.")
    else:
        st.write("No results found.")
"""

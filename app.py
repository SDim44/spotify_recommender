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
    selected_track_id = None
    if selected_track_info is not None:
        selected_track_id = track_ids[radio_options[selected_track_info]]
        # You can use the selected_track_id variable as needed
        st.write(f"Selected track ID: {selected_track_id}")

    # FEATURES FROM SPOTIFY
    if selected_track_id is not None:
        # FEATURES FROM SPOTIFY
        test_track_id = selected_track_id
        features = sp.audio_features(test_track_id)
        track_info = sp.track(test_track_id)
        artist_id = track_info['artists'][0]['id']
        artist_info = sp.artist(artist_id)
        features[0]['artists_genres'] = artist_info['genres']
        data_for_prediction = pd.DataFrame(features)

        st.write(data_for_prediction)

        # MODEL CALL
        model = SpotifyRecommender()
        model.load('./SpotifyRecommender_apiFeatures.pkl')
        output = model.predict(data_for_prediction, match_genre=True)
        #st.write(output.head(10))

        # LIST RECOMMENDED TRACKS
        # Will be filled by model
        track_ids = output["track_id"].tolist()

        # Set initial index for pagination
        start_index = 0

        def display_tracks(start, end):
            if track_ids:  # This checks if the track_ids list is not empty
                for track_id in track_ids[start:end]:
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

        # Initial display of tracks
        display_tracks(start_index, start_index+10)

        # Add button for loading more tracks
        if st.button('Load more'):
            # Increase the start_index by 10
            start_index += 10
            display_tracks(start_index, start_index+10)



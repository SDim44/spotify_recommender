import os
import pandas as pd
import ast
from SpotifyRecommender import SpotifyRecommender
from dagster import op,job,asset,repository

@op
def str_to_list(s): # Funktioniert in Dagster nicht
    try:
        s = str(s)[1:-1]
        if len(s) == 0:
            s = "'not defined'"
        items = s.split(',')
        return [item.replace("'","") for item in items]
    except (SyntaxError, ValueError):
        return ["not defined"]
    
def str_to_listV2(s):
    try:
        items = ast.literal_eval(s)
        if not isinstance(items, list):
            items = [items]
        return [str(item).strip("' ") for item in items]
    except (SyntaxError, ValueError):
        return ['not defined']

@asset
def import_albums():
    albums_path = os.path.join('spotify_data/Data Sources/spotify_albums.csv')
    return pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0').rename(columns=lambda x: "album_" + x)

@asset
def import_artists():
    albums_path = os.path.join('spotify_data/Data Sources/spotify_artists.csv')
    return pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0').rename(columns=lambda x: "artists_" + x)

@asset
def import_tracks():
    albums_path = os.path.join('spotify_data/Data Sources/spotify_tracks.csv')
    return pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0').rename(columns={'id':'track_id','artists_id':'track_artists_id'})

@asset
def import_lyrics_features():
    albums_path = os.path.join('spotify_data/Features Extracted/lyrics_features.csv')
    df = pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0')
    return df

@asset
def import_audio_features():
    albums_path = os.path.join('spotify_data/Features Extracted/low_level_audio_features.csv')
    df = pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0')
    return df

@op
def transform_albums(albums_raw):
    df = albums_raw.copy()
    df['album_available_markets'] = df['album_available_markets'].apply(lambda x: str_to_listV2(x))
    df['album_release_date'] = pd.to_datetime(df['album_release_date'])
    df['album_external_urls'] = df['album_external_urls'].apply(ast.literal_eval)
    df['album_images'] = df['album_images'].apply(ast.literal_eval)
    
    return df

@op
def transform_artists(artists_raw):
    df = artists_raw.copy()
    df['artists_genres'] = df['artists_genres'].apply(lambda x: str_to_listV2(x))

    return df

@op
def transform_tracks(tracks_raw):
    df = tracks_raw.copy()
    df['available_markets'] = df['available_markets'].apply(lambda x: str_to_listV2(x))
    df['track_artists_id'] = df['track_artists_id'].apply(lambda x: str_to_listV2(x))

    return df

@op
def matchall_spotify_data(tracks,albums,artists,audio_features,lyrics_features):
    tracks = tracks.explode('track_artists_id') # tack zeile pro artist

    merged = pd.merge(tracks, albums, left_on='album_id', right_on='album_id', how='inner')
    tracks_albums_artists = pd.merge(merged, artists, left_on='track_artists_id', right_on='artists_id', how='inner')
    # inner -> 101939 rows × 53 columns

    tracks_albums_artists_audio = pd.merge(tracks_albums_artists, audio_features, left_on='track_id', right_on='track_id', how='inner')
    tracks_albums_artists_audio_lyrics = pd.merge(tracks_albums_artists_audio, lyrics_features, left_on='track_id', right_on='track_id', how='inner')
    # inner -> 94924 rows × 266 columns

    return tracks_albums_artists_audio_lyrics

@op
def matchapi_spotify_data(tracks,albums,artists):
    tracks = tracks.explode('track_artists_id') # tack zeile pro artist

    merged = pd.merge(tracks, albums, left_on='album_id', right_on='album_id', how='inner')
    tracks_albums_artists = pd.merge(merged, artists, left_on='track_artists_id', right_on='artists_id', how='inner')
    # inner -> 101939 rows × 53 columns

    return tracks_albums_artists

@op
def train_fullmodel(df):
    model = SpotifyRecommender()
    model.train(df, 1000) # 1000 Zufällige Tracks aufwählen
    model.save("SpotifyRecommender_allFeatures_new.pkl")

@op
def train_apimodel(df):
    features = ['acousticness', 'danceability', 'duration_ms', 
            'energy', 'instrumentalness', 'key', 'liveness', 
            'loudness', 'mode', 'speechiness', 'tempo', 'time_signature']
    
    model = SpotifyRecommender()
    model.train(df, 0,features) # 0 für alle Tracks
    model.save("SpotifyRecommender_apiFeatures_new.pkl")


@job
def create_full_model():
    raw_albums = import_albums()
    raw_artists = import_artists()
    raw_tracks = import_tracks()
    raw_lyrics_features = import_lyrics_features()
    raw_audio_features = import_audio_features()

    albums = transform_albums(raw_albums)
    artists = transform_artists(raw_artists)
    tracks = transform_tracks(raw_tracks)
    lyrics_features = raw_lyrics_features
    audio_features = raw_audio_features
    
    df = matchall_spotify_data(tracks,albums,artists,audio_features,lyrics_features)
    
    train_fullmodel(df)


@job
def create_apiFeatures_model():
    raw_albums = import_albums()
    raw_artists = import_artists()
    raw_tracks = import_tracks()

    albums = transform_albums(raw_albums)
    artists = transform_artists(raw_artists)
    tracks = transform_tracks(raw_tracks)

    
    df = matchapi_spotify_data(tracks,albums,artists)
    
    train_apimodel(df)
    

@repository
def etl():
    return [create_apiFeatures_model, create_full_model]

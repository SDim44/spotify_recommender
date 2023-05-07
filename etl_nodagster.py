import os
import pandas as pd
import ast
import pickle
from datetime import datetime

workspacefolder = os.path.join('C:\\Users\\stefa\\OneDrive - FHWN\\Privat\\Studium\\MIT_2-Semester\\Analyseanwendungen\\spotify_recommander')

def str_to_list(s):
    s = str(s)[1:-1]
    if len(s) == 0:
        s = "'not defined'"
    items = s.split(',')
    return [item.replace("'","") for item in items]


def import_albums():
    albums_path = os.path.join(workspacefolder,'spotify_data/Data Sources/spotify_albums.csv')
    return pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0').rename(columns=lambda x: "album_" + x)

def import_artists():
    albums_path = os.path.join(workspacefolder,'spotify_data/Data Sources/spotify_artists.csv')
    return pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0').rename(columns=lambda x: "artists_" + x)

def import_tracks():
    albums_path = os.path.join(workspacefolder,'spotify_data/Data Sources/spotify_tracks.csv')
    return pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0').rename(columns={'id':'track_id','artists_id':'track_artists_id'})

def import_lyrics_features():
    albums_path = os.path.join(workspacefolder,'spotify_data/Features Extracted/lyrics_features.csv')
    df = pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0')
    return df

def import_audio_features():
    albums_path = os.path.join(workspacefolder,'spotify_data/Features Extracted/low_level_audio_features.csv')
    df = pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0')
    return df


def transform_albums(albums_raw):
    df = albums_raw.copy()
    df['album_available_markets'] = df['album_available_markets'].apply(lambda x: str_to_list(x))
    df['album_release_date'] = pd.to_datetime(df['album_release_date'])
    df['album_external_urls'] = df['album_external_urls'].apply(ast.literal_eval)
    df['album_images'] = df['album_images'].apply(ast.literal_eval)
    
    return df


def transform_artists(artists_raw):
    df = artists_raw.copy()
    df['artists_genres'] = df['artists_genres'].apply(lambda x: str_to_list(x))

    return df


def transform_tracks(tracks_raw):
    df = tracks_raw.copy()
    df['available_markets'] = df['available_markets'].apply(lambda x: str_to_list(x))
    df['track_artists_id'] = df['track_artists_id'].apply(lambda x: str_to_list(x))

    return df


def match_spotify_data(tracks,albums,artists,audio_features,lyrics_features):
    # tracks = tracks.explode('track_artists_id') # tack zeile pro artist

    merged = pd.merge(tracks, albums, left_on='album_id', right_on='album_id', how='inner')
    tracks_albums_artists = pd.merge(merged, artists, left_on='track_artists_id', right_on='artists_id', how='inner')
    # inner -> 101939 rows × 53 columns

    tracks_albums_artists_audio = pd.merge(tracks_albums_artists, audio_features, left_on='track_id', right_on='track_id', how='inner')
    tracks_albums_artists_audio_lyrics = pd.merge(tracks_albums_artists_audio, lyrics_features, left_on='track_id', right_on='track_id', how='inner')
    # inner -> 94924 rows × 266 columns

    return tracks_albums_artists_audio_lyrics


def track_info(df): # match_features
    df = df[['name','track_href','preview_url','analysis_url','href','lyrics','playlist','popularity','tempo','time_signature','track_id',
             'artists_name','artists_genres','artists_followers', 'artists_artist_popularity', 'artists_id',
             'album_name','album_release_date','album_images','album_total_tracks','album_external_urls','album_id']]
    return df


def save_pickl(model,name='model'):
    filename = f"{datetime.now().strftime('%Y-%m-%d')}_{name}.pickle"
    with open(filename,'wb') as f:
        f.dump(model,f)


def prepare_dataset():
    albums = transform_albums(import_albums())
    artists = transform_artists(import_artists())
    tracks = transform_tracks(import_tracks())
    lyrics_features = import_lyrics_features()
    audio_features = import_audio_features()
    data = match_spotify_data(tracks,albums,artists,audio_features,lyrics_features)
    return data
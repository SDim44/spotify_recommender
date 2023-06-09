
import os
import pandas as pd
import ast
import pickle
from datetime import datetime

from tests.knn import SpotifyRecommender

# @op
def str_to_list(s):
    s = str(s)[1:-1]
    if len(s) == 0:
        s = "'not defined'"
    items = s.split(',')
    return [item.replace("'","") for item in items]

# @asset
def import_albums():
    albums_path = os.path.join('spotify_data/Data Sources/spotify_albums.csv')
    return pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0').rename(columns=lambda x: "album_" + x)

# @asset
def import_artists():
    albums_path = os.path.join('spotify_data/Data Sources/spotify_artists.csv')
    return pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0').rename(columns=lambda x: "artists_" + x)

# @asset
def import_tracks():
    albums_path = os.path.join('spotify_data/Data Sources/spotify_tracks.csv')
    return pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0').rename(columns={'id':'track_id','artists_id':'track_artists_id'})

# @op
def import_lyrics_features():
    albums_path = os.path.join('spotify_data/Features Extracted/lyrics_features.csv')
    df = pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0')
    return df

# @op
def import_audio_features():
    albums_path = os.path.join('spotify_data/Features Extracted/low_level_audio_features.csv')
    df = pd.read_csv(albums_path,sep=',',index_col='Unnamed: 0')
    return df

# @op
def transform_albums(albums_raw):
    df = albums_raw.copy()
    df['album_available_markets'] = df['album_available_markets'].apply(lambda x: str_to_list(x))
    df['album_release_date'] = pd.to_datetime(df['album_release_date'])
    df['album_external_urls'] = df['album_external_urls'].apply(ast.literal_eval)
    df['album_images'] = df['album_images'].apply(ast.literal_eval)
    
    return df

# @op
def transform_artists(artists_raw):
    df = artists_raw.copy()
    df['artists_genres'] = df['artists_genres'].apply(lambda x: str_to_list(x))

    return df

# @op
def transform_tracks(tracks_raw):
    df = tracks_raw.copy()
    df['available_markets'] = df['available_markets'].apply(lambda x: str_to_list(x))
    df['track_artists_id'] = df['track_artists_id'].apply(lambda x: str_to_list(x))

    return df

# @asset
def match_spotify_data(tracks,albums,artists,audio_features,lyrics_features):
    tracks = tracks.explode('track_artists_id') # tack zeile pro artist

    merged = pd.merge(tracks, albums, left_on='album_id', right_on='album_id', how='inner')
    tracks_albums_artists = pd.merge(merged, artists, left_on='track_artists_id', right_on='artists_id', how='inner')
    # inner -> 101939 rows × 53 columns

    tracks_albums_artists_audio = pd.merge(tracks_albums_artists, audio_features, left_on='track_id', right_on='track_id', how='inner')
    tracks_albums_artists_audio_lyrics = pd.merge(tracks_albums_artists_audio, lyrics_features, left_on='track_id', right_on='track_id', how='inner')
    # inner -> 94924 rows × 266 columns

    return tracks_albums_artists_audio_lyrics
    

# @op
def track_info(df): # match_features
    df = df[['name','track_href','preview_url','analysis_url','href','lyrics','playlist','popularity','tempo','time_signature','track_id',
             'artists_name','artists_genres','artists_followers', 'artists_artist_popularity', 'artists_id',
             'album_name','album_release_date','album_images','album_total_tracks','album_external_urls','album_id']]
    return df

# @op
def save_pickl(model,name='model'):
    filename = f"{datetime.now().strftime('%Y-%m-%d')}_{name}.pickle"
    with open(filename,'wb') as f:
        f.dump(model,f)


def save_dataset(df):
    path = 'evaluation/match_spotify_data.pickle'
    df.to_pickle(path)
    return True


def prepare_dataset():
    albums = transform_albums(import_albums())
    artists = transform_artists(import_artists())
    tracks = transform_tracks(import_tracks())
    lyrics_features = import_lyrics_features()
    audio_features = import_audio_features()
    df = match_spotify_data(tracks,albums,artists,audio_features,lyrics_features)
    save_dataset(df)
    return df


# # @job
# def save_dataset(data=prepare_dataset()):
#     dataset_path = 'evaluation/match_spotify_data.pickle'
#     data.to_pickle(dataset_path)

# # @job
# def load_model(df = prepare_dataset()):
#     track_infos = track_info(df)

#     save_pickl(track_infos,'track_info')
    

# # @job
# def train_model(df = prepare_dataset()):
#     try:
#         model = SpotifyRecommender().load('model.pickle')
#     except:
#         model = SpotifyRecommender()
        
#     model.train(df)
#     model.save(f"{datetime.now().strftime('%Y-%m-%d')}_model.pickle")


if __name__ == '__main__':
    pass
    # train_model()

if __name__ == '__main__':
    prepare_dataset()
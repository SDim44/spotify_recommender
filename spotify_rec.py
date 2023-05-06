
import os
import pandas as pd
from dagster import op,job,get_dagster_logger
log = get_dagster_logger()


def import_albums():
    albums_path = os.path.join('spotify_data/Data Sources/spotify_albums.csv')
    df = pd.read_csv(albums_path,sep=',')
    return df

def import_artists():
    albums_path = os.path.join('spotify_data/Data Sources/spotify_artists.csv')
    df = pd.read_csv(albums_path,sep=',')
    return df

def import_tracks():
    albums_path = os.path.join('spotify_data/Data Sources/spotify_tracks.csv')
    df = pd.read_csv(albums_path,sep=',')
    return df

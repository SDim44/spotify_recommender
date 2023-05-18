import pandas as pd
import numpy as np
import pickle
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import NearestNeighbors


class SpotifyRecommender:
    supported = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence'] + [f'Chroma_{i}' for i in range(1, 13)] + [f'MEL_{i}' for i in range(1, 129)] + [f'MFCC_{i}' for i in range(1, 49)] + [f'Spectral_contrast_{i}' for i in range(1, 8)] + [f'Tonnetz_{i}' for i in range(1, 7)] + ['ZCR', 'entropy_energy', 'spectral_bandwith', 'spectral_centroid', 'spectral_rollOff_max', 'spectral_rollOff_min']
    return_columns = ['name','artists_name','artists_genres','album_name','track_href','preview_url','analysis_url','href','lyrics','playlist','popularity','tempo','time_signature','track_id',
             'artists_followers', 'artists_artist_popularity', 'artists_id','album_release_date','album_images','album_total_tracks','album_external_urls','album_id']
    def __init__(self,load_model="",k=20):
        self.scaler = None
        self.pca = None
        self.sfm = None
        self.knn = None
        self.k = k
        self.filepath = 'saved_modelV2.pkl'
        self.dataset = None
        self.trained_features = []
        if not load_model == "":
            path = os.path.join(load_model)
            self.load(path)
        

    def get_features(self,data,features):
        listed = data.columns.tolist()
        for f in listed:
            if f in self.supported and f in features:
                self.trained_features.append(f)
        return


    def train(self,dataset,n_lines=0,features=supported):
        self.get_features(dataset,features)
        self.dataset = dataset

        if not n_lines == 0:
            dataset = dataset.sample(n=n_lines)

        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(dataset[self.trained_features])

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(scaled_data, dataset.index)  # Hier verwende ich data.index anstelle von target

        self.sfm = SelectFromModel(rf, threshold='mean')
        self.sfm.fit(scaled_data, dataset.index)  # Hier verwende ich data.index anstelle von target

        self.pca = PCA(n_components=0.95)
        reduced_data = self.pca.fit_transform(scaled_data[:, self.sfm.get_support()])

        self.knn = NearestNeighbors(n_neighbors=self.k)
        self.knn.fit(reduced_data)

    def predict(self, df_selected,k=None):
        if not k==None:
            self.knn.n_neighbors = k

        new_scaled_data = self.scaler.transform(df_selected[self.trained_features])
        new_reduced_data = self.pca.transform(new_scaled_data[:, self.sfm.get_support()])
        distances, indices = self.knn.kneighbors(new_reduced_data)
        
        self.knn.n_neighbors = self.k
        return self.dataset[self.return_columns].iloc[indices[0]]#, distances, indices

    def save(self, file_path=None):
        if file_path == None:
            file_path = self.filepath
        model_data = {
            'knn': self.knn,
            'scaler': self.scaler,
            'pca': self.pca,
            'sfm': self.sfm,
            'dataset': self.dataset,
            'filepath': file_path,
            'trained_features': self.trained_features
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        self.knn = model_data['knn']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.sfm = model_data['sfm']
        self.filepath = model_data['filepath']
        self.trained_features =  model_data['trained_features']
        self.dataset = model_data['dataset']

if __name__ == '__main__':
    import tests.etl_test as etl
    data = etl.prepare_dataset()

    features = ['acousticness', 'danceability', 'duration_ms', 
                'energy', 'instrumentalness', 'key', 'liveness', 
                'loudness', 'mode', 'speechiness', 'tempo', 'time_signature']
    model = SpotifyRecommender()
    model.train(data, 5000, features)

    model.save('tests/model_test.pkl')

    model.predict(model.dataset[model.dataset['track_id']=='0EYOdF5FCkgOJJla8DI2Md'])
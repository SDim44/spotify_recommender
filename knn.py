import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import NearestNeighbors


class SpotifyRecommander:
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.sfm = None
        self.knn = None
        self.filepath = None
        self.features = []

    def get_features(self,data):
        supported = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence'] + [f'Chroma_{i}' for i in range(1, 13)] + [f'MEL_{i}' for i in range(1, 129)] + [f'MFCC_{i}' for i in range(1, 49)] + [f'Spectral_contrast_{i}' for i in range(1, 8)] + [f'Tonnetz_{i}' for i in range(1, 7)] + ['ZCR', 'entropy_energy', 'spectral_bandwith', 'spectral_centroid', 'spectral_rollOff_max', 'spectral_rollOff_min']
        listed = data.columns.tolist()
        for f in listed:
            if f in supported:
                self.features.append(f)


    def train(self, data):
        self.features = self.get_features(data)

        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(data[self.features])

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(scaled_data, data.index)  # Hier verwende ich data.index anstelle von target

        self.sfm = SelectFromModel(rf, threshold='mean')
        self.sfm.fit(scaled_data, data.index)  # Hier verwende ich data.index anstelle von target

        self.pca = PCA(n_components=0.95)
        reduced_data = self.pca.fit_transform(scaled_data[:, self.sfm.get_support()])

        self.knn = NearestNeighbors(n_neighbors=5)
        self.knn.fit(reduced_data)

    def predict(self, new_data):
        new_scaled_data = self.scaler.transform(new_data[self.features])
        new_reduced_data = self.pca.transform(new_scaled_data[:, self.sfm.get_support()])
        distances, indices = self.knn.kneighbors(new_reduced_data)
        return distances, indices

    def save(self, file_path=None):
        if file_path == None:
            file_path = self.filepath
        model_data = {
            'knn': self.knn,
            'scaler': self.scaler,
            'pca': self.pca,
            'sfm': self.sfm,
            'filepath': file_path,
            'features': self.features
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
        self.filepath = file_path
        self.features =  model_data['features']
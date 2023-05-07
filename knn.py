import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import NearestNeighbors

class SpotifyRecommander:
    supported = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence'] + [f'Chroma_{i}' for i in range(1, 13)] + [f'MEL_{i}' for i in range(1, 129)] + [f'MFCC_{i}' for i in range(1, 49)] + [f'Spectral_contrast_{i}' for i in range(1, 8)] + [f'Tonnetz_{i}' for i in range(1, 7)] + ['ZCR', 'entropy_energy', 'spectral_bandwith', 'spectral_centroid', 'spectral_rollOff_max', 'spectral_rollOff_min']
    def __init__(self,dataset,features=supported,k=20):
        self.scaler = None
        self.pca = None
        self.sfm = None
        self.knn = None
        self.k = k
        self.filepath = None
        self.dataset = dataset
        self.features = []
        self.get_features(dataset,features)

    def get_features(self,data,features):
        listed = data.columns.tolist()
        for f in listed:
            if f in self.supported and f in features:
                self.features.append(f)


    def train(self):

        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(self.dataset[self.features])

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(scaled_data, self.dataset.index)  # Hier verwende ich data.index anstelle von target

        self.sfm = SelectFromModel(rf, threshold='mean')
        self.sfm.fit(scaled_data, self.dataset.index)  # Hier verwende ich data.index anstelle von target

        self.pca = PCA(n_components=0.95)
        reduced_data = self.pca.fit_transform(scaled_data[:, self.sfm.get_support()])

        self.knn = NearestNeighbors(n_neighbors=self.k)
        self.knn.fit(reduced_data)

    def predict(self, track_id):
        new_data = self.dataset[self.dataset['track_id']==track_id]
        new_scaled_data = self.scaler.transform(new_data[self.features])
        new_reduced_data = self.pca.transform(new_scaled_data[:, self.sfm.get_support()])
        distances, indices = self.knn.kneighbors(new_reduced_data)
        
        return self.dataset.iloc[indices[0]]#, distances, indices

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
        self.filepath = model_data['file_path']
        self.features =  model_data['features']

if __name__ == '__main__':
    import etl_nodagster as etl
    from datetime import datetime

    t = datetime.now().strftime('%Y-%m-%d')
    
    data = etl.prepare_dataset()


    # Wenn keine features übergeben werden, wird das modell anhand aller features erstellt
    # features = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature']

    model = SpotifyRecommander(data)#,features)
    # model.dataset
    model.train()
    model.save(f"{t}_testmodel.pickle")

    out = model.predict('0EYOdF5FCkgOJJla8DI2Md')
    print(out[['name','artists_name','album_name','artists_genres']])

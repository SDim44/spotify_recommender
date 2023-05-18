import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

class SpotifyRecommender:
    supported_values = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence'] + [f'Chroma_{i}' for i in range(1, 13)] + [f'MEL_{i}' for i in range(1, 129)] + [f'MFCC_{i}' for i in range(1, 49)] + [f'Spectral_contrast_{i}' for i in range(1, 8)] + [f'Tonnetz_{i}' for i in range(1, 7)] + ['ZCR', 'entropy_energy', 'spectral_bandwith', 'spectral_centroid', 'spectral_rollOff_max', 'spectral_rollOff_min']
    supported_categories = ['artists_genres']
    evaluation_result = []
    def __init__(self,k=20):
        self.scaler = None
        self.pca = None
        self.sfm = None
        self.knn = None
        self.k = k
        self.filepath = None
        self.dataset = pd.DataFrame()
        self.features = None
        self.best_features = ['acousticness','danceability','duration_ms','energy','instrumentalness',
                            'liveness','loudness','speechiness','tempo','valence','Chroma_1',
                            'Chroma_2','Chroma_3','Chroma_4','Chroma_5','Chroma_6','Chroma_7',
                            'Chroma_8','Chroma_9','Chroma_10','Chroma_11','Chroma_12','MEL_1','MEL_2',
                            'MEL_3','MEL_4','MEL_5','MEL_6','MEL_7','MEL_8','MEL_13','MEL_14',
                            'MEL_16','MEL_17','MEL_18','MEL_19','MEL_20','MEL_22','MEL_23','MEL_24',
                            'MEL_27','MEL_30','MEL_51','MFCC_2','MFCC_3','MFCC_4','MFCC_5','MFCC_6',
                            'MFCC_7','MFCC_8','MFCC_9','MFCC_10','MFCC_11','MFCC_12','MFCC_13',
                            'MFCC_14','MFCC_15','MFCC_16','MFCC_17','MFCC_18','MFCC_19','MFCC_20',
                            'MFCC_21','MFCC_22','MFCC_23','MFCC_24','MFCC_25','MFCC_26','MFCC_27',
                            'MFCC_28','MFCC_29','MFCC_30','MFCC_31','MFCC_32','MFCC_33','MFCC_34',
                            'MFCC_35','MFCC_36','MFCC_37','MFCC_38','MFCC_39','MFCC_40','MFCC_41',
                            'MFCC_42','MFCC_43','MFCC_44','MFCC_45','MFCC_46','MFCC_47','MFCC_48',
                            'Spectral_contrast_1','Spectral_contrast_2','Spectral_contrast_3',
                            'Spectral_contrast_4','Spectral_contrast_5','Spectral_contrast_6',
                            'Spectral_contrast_7','Tonnetz_1','Tonnetz_2','Tonnetz_3','Tonnetz_4',
                            'Tonnetz_5','Tonnetz_6','entropy_energy','spectral_centroid',
                            'spectral_rollOff_min']
        self.reduced_data = None
        self.encoded_data = None
        self.feature_df = None
        self.trained_values = []
        self.trained_categories = []
        
    def __get_features_in_dataset(self,df):
        listed = df.columns.tolist()
        new_featurelist = []
        for f in listed:
            if f in self.supported_values:
                new_featurelist.append(f)
        
        self.features = new_featurelist

    def import_dataset(self,df):
        self.dataset = df
        self.__get_features_in_dataset(df)


    def evaluate_features_in_dataset(self):
        '''
        Duration on a Dell G5 Notebook
        Scale: 0.3 sec
        RandomForestRegressor: 71 min
        SelectFromModel: 72 min
        PCA: 0.8 sec
        '''
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(self.dataset[self.features])

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(scaled_data, self.dataset.index) 

        self.sfm = SelectFromModel(rf, threshold='mean')
        self.sfm.fit(scaled_data, self.dataset.index) 

        self.pca = PCA(n_components=0.95)
        self.reduced_data = self.pca.fit_transform(scaled_data[:, self.sfm.get_support()])

        self.best_features = self.dataset[self.features].columns[self.sfm.get_support()].tolist()

        
    def binarize_categories(self,dataset):
        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(dataset)
        df = pd.DataFrame(encoded, columns=mlb.classes_)
        return df
    
    def __add_missing_columns(df, column_list):
        for col in column_list:
            if col not in df.columns:
                df[col] = 0
        return df

    def create_model(self):
        '''KNN: 18.5 sec'''
        print(f'---- Creating model ----')
        self.knn = NearestNeighbors()
        self.trained_values = self.best_features

        if len(self.trained_categories) > 0:
            self.trained_categories = self.supported_categories
            encoded_df = self.binarize_categories(self.dataset['artists_genres'])
            reduced_df = pd.DataFrame(self.reduced_data, columns=[f"PC{i+1}" for i in range(self.reduced_data.shape[1])])
            self.feature_df = reduced_df.join(encoded_df)
            self.knn.fit(self.feature_df)
            
        else:
            self.knn.fit(self.reduced_data)
            

    def get_neighbors(self, predict_df, k=None):
        if k == None:
            k = self.k
        self.knn.n_neighbors = k
        self.features = self.__get_features_in_dataset(predict_df)
        value_df= predict_df[self.supported_values]
        new_scaled_data = self.scaler.transform(value_df)
        new_reduced_data = self.pca.transform(new_scaled_data[:, self.sfm.get_support()])

        if len(self.trained_categories) > 0:
            reduced_df = pd.DataFrame(new_reduced_data, columns=[f"PC{i+1}" for i in range(new_reduced_data.shape[1])])
            category_df = predict_df[self.trained_categories]
            encoded_df = self.binarize_categories(predict_df['artists_genres'])
            encoded_df = __add_missing_columns(encoded_df,self.trained_categories)

            to_predict = reduced_df.join(encoded_df)
        else:
            to_predict = new_reduced_data

        distances, indices = self.knn.kneighbors(to_predict)
        
        return self.dataset.iloc[indices[0]]

    def save(self, file_path=None):
        if file_path == None:
            file_path = self.filepath
        model_data = {
            'knn': self.knn,
            'scaler': self.scaler,
            'pca': self.pca,
            'sfm': self.sfm,
            'filepath': file_path,
            'trained_values':self.trained_values,
            'trained_categories':self.trained_categories
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
        self.trained_values = model_data['trained_values']
        self.trained_categories = model_data['trained_categories']


if __name__ == '__main__':
    file_path=r'evaluation/SpotifyRecommenderV1.pickle'

    df = pd.read_pickle(r'evaluation/match_spotify_data.pickle')
    sp = SpotifyRecommender()
    sp.load(file_path)

    predata = df[df['track_id']=='0EYOdF5FCkgOJJla8DI2Md']
    print(predata['artists_genres'])

    results = sp.get_neighbors(predata)
    print(results)

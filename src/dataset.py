import torch
import pandas as pd

class MovieDataset:
    def __init__(self, users, movies, ratings):
        ratings.reset_index(inplace=True)
        merged = pd.merge(ratings, users, left_on='user', right_on='id').drop('id', axis=1).set_index('index')
        merged = pd.merge(merged, movies, left_on='movie', right_on='id').drop('id', axis=1)
    
        self.data = merged.sort_index()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        user = self.data['user'].iloc[item]
        rating = self.data['rating'].iloc[item]
        movie = self.data['movie'].iloc[item]
        rest = self.data.iloc[item].drop([ 'user', 'rating', 'movie'])
        return {
            'user': torch.tensor(user, dtype = torch.long),
            'movie': torch.tensor(movie, dtype = torch.long),
            'rest': torch.tensor(rest, dtype = torch.float),
            'rating': torch.tensor(rating, dtype = torch.float)
        }
    
    def get_user_data(self, user):
        data = self.data[self.data['user'] == user]
        user = data['user']
        rating = data['rating']
        movie = data['movie']
        rest = data.drop(['user', 'rating', 'movie'], axis = 1)
        rest = rest.astype(float)
        return {
            'user': torch.tensor(user.values, dtype = torch.long),
            'movie': torch.tensor(movie.values, dtype = torch.long),
            'rest': torch.tensor(rest.values, dtype = torch.float),
            'rating': torch.tensor(rating.values, dtype = torch.float)
        }


    def return_data(self):
        return self.data



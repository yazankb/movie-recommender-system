import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from dataset import MovieDataset
from torch.utils.data import DataLoader
from model import RecSysModel
import optuna
import logging
from tqdm import tqdm
import sys
sys.path.append('..')
from benchmark.evaluate import validate_model


data_relative_path = "./../data/raw/ml-100k/u.data"
moives_relative_path = "./../data/raw/ml-100k/u.item"
user_relative_path = "./../data/raw/ml-100k/u.user"

num_users = 944
num_movies = 1683

column_names = ['user', 'movie', 'rating', 'time']
users_columns_names = ["id", "age", "gender", "occupation", "zip code"]
movies_columns_names = ["id",
                        "name",
                        "date",
                        "empty",
                        "url",
                        "unknown",
                        "Action",
                        "Adventure",
                        "Animation",
                        "Children's",
                        "Comedy",
                        "Crime",
                        "Documentary",
                        "Drama",
                        "Fantasy",
                        "Film-Noir",
                        "Horror",
                        "Musical",
                        "Mystery",
                        "Romance",
                        "Sci-Fi",
                        "Thriller",
                        "War",
                        "Western"]


if torch.cuda.is_available():
    device = torch.device('cuda')  # Use GPU if available
else:
    device = torch.device('cpu')  # Use CPU if GPU is not available


def train(model, criterion, optimizer, scheduler, train_dataloader, epochs = 5):
    
    model.train()
    # Train the model
    for epoch in tqdm(range(epochs)):
        for user_data in train_dataloader:
            user = user_data['user']
            movie = user_data['movie']
            rest = user_data['rest']
            rating = user_data['rating']
            
            # Move data to device
            user, movie, rest, rating = user.to(device), movie.to(device), rest.to(device), rating.to(device)
            # Forward pass
            outputs = model(user, movie, rest)
            loss = criterion(outputs, rating)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

        # Update scheduler
        scheduler.step()



def objective(trial):

    # Define hyperparameters using trial.suggest_ methods
    embedding_size = trial.suggest_int('embedding_size', 20, 100)
    hidden_layers = trial.suggest_categorical('hidden_layers', [(64, 32), (128, 64), (256, 128, 64)])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

    
    train_df = pd.read_csv(data_relative_path + f'/../ua.base', sep = '\t', header = None, names = column_names).drop('time', axis = 1)
    val_df = pd.read_csv(data_relative_path + f'/../ua.test', sep = '\t', header = None, names = column_names).drop('time', axis = 1)
    
    users_df = pd.get_dummies(pd.read_csv(user_relative_path, sep = "|", header = None, names = users_columns_names).drop('zip code' , axis = 1))
    users_df['age'] = users_df['age'].apply(lambda x: (x-users_df['age'].mean())/ users_df['age'].std())
    movies_df = pd.read_csv(moives_relative_path, sep = '|', header = None, encoding='ISO-8859-1', names = movies_columns_names).drop(['name', 'date', 'empty', 'url'], axis = 1 )
    
    
    train_dataset = MovieDataset(users_df, movies_df , train_df)
    val_dataset = MovieDataset(users_df, movies_df , val_df)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True) 

    # Initialize model, loss, optimizer, scheduler
    model = RecSysModel(num_users, num_movies, embedding_size, hidden_layers)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    train(model, criterion, optimizer, scheduler, train_dataloader)
    rmse_loss, dcg_score = validate_model(model, val_dataset, 943)

    return rmse_loss


if __name__ == '__main__':
    optuna.logging.get_logger("optuna").setLevel(logging.INFO)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=12)

    # After finding the best hyperparameters:
    best_params = study.best_params
    print(best_params)
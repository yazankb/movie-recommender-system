import torch
import torch.nn as nn
import torch.nn.functional as F


class RecSysModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size, hidden_layers=[64, 32]):
        super(RecSysModel, self).__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)

        # Neural network layers
        self.hidden_layers = nn.ModuleList()
        input_size = 2 * embedding_size + 43  # Concatenated user and movie embeddings plus the additional information

        for hidden_size in hidden_layers:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(input_size, 1)

    def forward(self, user, movie, rest):
        # Get embeddings
        user_embedded = self.user_embedding(user)
        movie_embedded = self.movie_embedding(movie)

        # Concatenate user and movie embeddings
        combined = torch.cat([user_embedded, movie_embedded, rest], dim=1)

        # Forward pass through hidden layers with ReLU activation
        for layer in self.hidden_layers:
            combined = F.relu(layer(combined))

        # Output layer
        rating_prediction = self.output_layer(combined)
        return rating_prediction.squeeze()  
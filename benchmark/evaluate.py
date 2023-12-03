import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import dcg_score


def validate_model(model, valid_dataset, num_users):
    """
    Validates a PyTorch model using a validation dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to be validated.
        valid_dataset: The validation dataset.
        num_users: The number of users

    Returns:
        avg_loss_rmse (float): The average root mean squared error (RMSE) loss.
        avg_dcg (float): The average discounted cumulative gain (DCG) score.

    """
    print("Model Validation Started")
    avg_loss_rmse = 0  # Initialize average RMSE loss
    total_loss_mse = 0  # Initialize total MSE loss
    avg_dcg = 0  # Initialize average DCG score
    dcg_total = 0  # Initialize total DCG score
    lens = 0  # Initialize the total number of ratings

    if torch.cuda.is_available():
        device = torch.device('cuda')  # Use GPU if available
    else:
        device = torch.device('cpu')  # Use CPU if GPU is not available

    model.to(device)  # Move the model to the selected device
    model.eval()  # Set the model to evaluation mode
    x = 0
    for i in tqdm(range(1, num_users+1)):  # Loop through user data 
        user_data = valid_dataset.get_user_data(i)  # Get user data from the validation dataset
        user = user_data['user']
        movie = user_data['movie']
        rest = user_data['rest']
        rating = user_data['rating']
        lens += len(rating)  # Increment the total number of ratings

        if len(user) == 0:
            continue  # Skip if there is no user data for this user

        user, movie, rest, rating = user.to(device), movie.to(device), rest.to(device), rating.to(device)

        outputs = model(user, movie, rest)  # Forward pass through the model

        mse_loss = nn.MSELoss(reduction='sum')(outputs, rating)  # Calculate MSE loss
        total_loss_mse += mse_loss.item()

        # one rating ignore
        if len(user) != 1:
            dcg_total += dcg_score(rating.detach().cpu().unsqueeze(0), outputs.detach().cpu().unsqueeze(0))
             
    avg_loss_rmse = np.sqrt(total_loss_mse / lens)  # Calculate average RMSE
    avg_dcg = dcg_total / num_users  # Calculate average DCG score

    return avg_loss_rmse, avg_dcg
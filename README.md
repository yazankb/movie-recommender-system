# Movie Recommender System
## Yazan Kbaili
## y.kbaili@innopolis.university

This is a Movie Recommender System project that utilizes collaborative filtering and deep learning techniques to recommend movies to users. It is designed to work with the MovieLens 100K dataset.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)

## Overview

This project aims to build a movie recommender system using collaborative filtering techniques and deep learning models. It includes components such as data preprocessing, model training, hyperparameter optimization, and evaluation.

## Project Structure

The project is structured as follows:
- `data/`
  - `external/` # Data from third-party sources
  - `interim/` # Intermediate data that has been transformed
  - `raw/` # The original, immutable data dump
- `models/` # Trained and serialized models, model checkpoints
- `notebooks/` # Jupyter notebooks for exploratory data analysis and model development
- `references/` # Data dictionaries, manuals, and all explanatory materials
- `reports/`
  - `figures/` # Generated graphics and figures to be used in reporting
  - `final_report.pdf` # Compiled report detailing explorations, methodologies, and results
- `benchmark/`
  - `data/` # Dataset used for model evaluation
  - `evaluate.py` # Evaluation script for the trained models


## Getting Started

To get started with this project, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Download the MovieLens 100K dataset and place it in the appropriate directory under the `data/raw` folder.
4. Train and evaluate the recommender system by running the main training script.

## Usage

To train the recommender system and optimize hyperparameters, use the `train.py` script. You can specify different hyperparameters and settings within this script. It includes the following key components:
1. Data loading and preprocessing from CSV files.
2. Model training using PyTorch, including the definition of the neural network architecture.
3. Hyperparameter optimization using Optuna.
4. Logging and monitoring training progress.
5. Saving the best model and hyperparameters.

```bash
python train.py

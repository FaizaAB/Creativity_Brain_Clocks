# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:54:06 2023

@author: Carlos Coronel
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import bct
from plot_violins import violin_plot
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
import nibabel as nb
from wordcloud import WordCloud
from brainsmash.mapgen.base import Base
from scipy.spatial.distance import pdist, squareform
from neuromaps.datasets import fetch_fslr
from surfplot import Plot
import time
import pandas as pd
import warnings
import os
import sys
warnings.filterwarnings('ignore') 

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

log_file = open(os.path.join(RESULTS_DIR, "run_log_CNN.txt"), "w", encoding="utf-8", buffering=1)
sys.stdout = Tee(sys.stdout, log_file)
print(f"Saving results to: {os.path.abspath(RESULTS_DIR)}")

def _sanitize_filename(name):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")

def save_figure(filename, fig=None, dpi=300):
    if fig is None:
        fig = plt.gcf()
    if not filename.lower().endswith(".png"):
        filename = f"{filename}.png"
    base, ext = os.path.splitext(_sanitize_filename(filename))
    filepath = os.path.join(RESULTS_DIR, f"{base}_CNN{ext}")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {filepath}")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


class CNNRegressor(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x).squeeze(-1)


def standardize_features(train_array, arrays_to_transform):
    mean = train_array.mean(axis=0, keepdims=True)
    std = train_array.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    transformed = [((arr - mean) / std).astype(np.float32) for arr in arrays_to_transform]
    return transformed, mean.astype(np.float32), std.astype(np.float32)


def predict_with_cnn(model, x_array, batch_size=256):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x_array), batch_size):
            batch = torch.from_numpy(x_array[start:start + batch_size]).unsqueeze(1).to(DEVICE)
            preds = model(batch).detach().cpu().numpy()
            outputs.append(preds)
    return np.concatenate(outputs, axis=0)


def train_cnn_regressor(x_train_full, y_train_full, seed, epochs=40, batch_size=64, learning_rate=1e-3, weight_decay=1e-4):
    set_seed(seed)
    # internal validation split
    n = len(x_train_full)
    val_size = max(1, int(round(0.1 * n)))
    perm = np.random.RandomState(seed).permutation(n)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    if len(train_idx) == 0:
        train_idx = val_idx
        val_idx = perm[:1]

    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]

    model = CNNRegressor(x_train_full.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train).unsqueeze(1), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=min(batch_size, len(x_train)), shuffle=True
    )
    val_x = torch.from_numpy(x_val).unsqueeze(1).to(DEVICE)
    val_y = torch.from_numpy(y_val.astype(np.float32)).to(DEVICE)

    best_state = None
    best_val = float('inf')
    patience = 8
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(val_x)
            val_loss = criterion(val_preds, val_y).item()
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_val

def get_uptri(x):
    """
    Extracts the upper triangular (excluding diagonal) elements of a square matrix `x` 
    and returns them as a 1D vector.
    
    Parameters:
        x (ndarray): A square matrix (n x n).
    
    Returns:
        vector (ndarray): A 1D array containing the upper triangular elements of `x` 
                          (above the diagonal), in row-wise order.
    """

def cohen_d(x,y):
    """
    Function for computing Cohen's D effect size.
    
    Inputs:
        x,y: numpy arrays, vectors with observations.
    Output:
        Effect size.
    
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x) ** 2 + (ny-1)*np.std(y) ** 2) / dof)

# Define a helper function for printing formatted results
def print_stats(group_name, group1, group2, paired=False):
    if paired:
        t_stat, p_val = stats.ttest_rel(group1, group2)
    else:
        t_stat, p_val = stats.ttest_ind(group1, group2)
    d = cohen_d(group1, group2)
    delta = np.mean(group1) - np.mean(group2)
    print(f"{group_name}, Delta BAGs = {delta:.4f}, t = {t_stat:.4f}, p = {p_val:.4f}, d = {d:.4f}")


def print_corr(label, x, y):
    try:
        r, p = stats.pearsonr(x, y)
        print(f"{label}, r = {r:.4f}, p = {p:.4f}")
        return p
    except Exception as e:
        print(f"{label}, correlation failed: {e}")
        return None

# Define a helper function for printing paired test results
def print_paired(label, x, y):
    t_stat, p_val = stats.ttest_rel(x, y)
    d = cohen_d(x, y)
    delta = np.mean(x) - np.mean(y)
    print(f"{label}, Delta = {delta:.4f}, t = {t_stat:.4f}, p = {p_val:.4f}, d = {d:.4f}")


# Helper to print Benjamini-Hochberg adjusted p-values
def print_corrected_pvals(labels, pvals):
    for label, p in zip(labels, pvals):
        print(f"{label} (FDR adjusted), p = {p:.4f}")


# Function to set limits and ticks for a plot manually
def set_limits_and_ticks(ax, x_low, x_high, y_low, y_high):
    x_range = x_high - x_low  # Calculate x-axis range
    y_range = y_high - y_low  # Calculate y-axis range

    # Extend limits slightly beyond data range for clarity
    x_lim = [x_low - 0.05 * x_range, x_high + 0.05 * x_range]
    y_lim = [y_low - 0.05 * y_range, y_high + 0.05 * y_range]

    # Generate 4 ticks for both x and y axes within the specified limits
    x_ticks = np.linspace(x_low, x_high, 4)
    y_ticks = np.linspace(y_low, y_high, 4)

    # Apply limits and ticks to the plot
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    

#%%
# Data for training the SVMs (1240 participants + data augmentation)
input_FCs_1 = np.load('Training_SVMs_Data/FCs_training_augmented_1.npy')
input_FCs_2 = np.load('Training_SVMs_Data/FCs_training_augmented_2.npy')
input_FCs_3 = np.load('Training_SVMs_Data/FCs_training_augmented_3.npy')
input_FCs_4 = np.load('Training_SVMs_Data/FCs_training_augmented_4.npy')
input_FCs = np.concatenate((input_FCs_1, input_FCs_2, input_FCs_3, input_FCs_4), axis = 2)
input_ages = np.load('Training_SVMs_Data/ages_training_augmented.npy')

# Tango
ages_high_tango = np.load("Tango/ages_high_tango.npy")
ages_low_tango = np.load("Tango/ages_low_tango.npy")
FCs_high_tango = np.load("Tango/FCs_high_tango.npy")
FCs_low_tango = np.load("Tango/FCs_low_tango.npy")
hours_high_tango = np.load("Tango/hours_high_tango.npy")
hours_low_tango = np.load("Tango/hours_low_tango.npy")

# Gaming
ages_SC1 = np.load("Gaming/ages_SC1.npy")
ages_SC2 = np.load("Gaming/ages_SC2.npy")
FCs_SC1 = np.load("Gaming/FCs_SC1.npy")
FCs_SC2 = np.load("Gaming/FCs_SC2.npy")
playing_time = np.load("Gaming/playing_time.npy")

# Visual
ages_visual = np.load("Visual/ages_visual.npy")
ages_nonvisual = np.load("Visual/ages_nonvisual.npy")
FCs_nonvisual = np.load("Visual/FCs_nonvisual.npy")
FCs_visual = np.load("Visual/FCs_visual.npy")
experience_visual = np.load("Visual/experience_visual.npy")

# Learning
ages_sonata = np.load("Learning/ages_sonata.npy")
APM_post = np.load("Learning/APM_post.npy")
APM_pre = np.load("Learning/APM_pre.npy")
FCs_sonata_post = np.load("Learning/FCs_sonata_post.npy")
FCs_sonata_pre = np.load("Learning/FCs_sonata_pre.npy")

# Active control
ages_sonata_active = np.load("Learning/ages_sonata_active.npy")
FCs_sonata_post_active = np.load("Learning/FCs_sonata_post_active.npy")
FCs_sonata_pre_active = np.load("Learning/FCs_sonata_pre_active.npy")


#%%

#just a dummy matrix
new_ROIs = input_FCs.shape[0]
dummy_mat = np.zeros((new_ROIs,new_ROIs))
for i in range(0,new_ROIs-1):
    for j in range(1+i,new_ROIs):
        dummy_mat[i,j] = 1
triu_idx = dummy_mat == 1
new_pairs = np.sum(triu_idx)

#vectorizing FCs
vectorized = (input_FCs[triu_idx,:].T).T
vectorized_high_tango = (FCs_high_tango[triu_idx, :].T).T
vectorized_low_tango = (FCs_low_tango[triu_idx, :].T).T
vectorized_SC1 = (FCs_SC1[triu_idx, :].T).T
vectorized_SC2 = (FCs_SC2[triu_idx, :].T).T
vectorized_nonvisual = (FCs_nonvisual[triu_idx, :].T).T
vectorized_visual = (FCs_visual[triu_idx, :].T).T	
vectorized_sonata_post = (FCs_sonata_post[triu_idx, :].T).T
vectorized_sonata_pre = (FCs_sonata_pre[triu_idx, :].T).T
vectorized_sonata_post_active = (FCs_sonata_post_active[triu_idx, :].T).T
vectorized_sonata_pre_active = (FCs_sonata_pre_active[triu_idx, :].T).T


#%%

# Number of repetitions for CNN
reps = 15

# Arrays to save results across repetitions
rreps = np.zeros(reps)  # Stores correlation results for each repetition
ereps = np.zeros(reps)  # Stores mean absolute error for each repetition
rmse_reps = np.zeros(reps)  # Stores RMSE for each repetition
r2_reps = np.zeros(reps)  # Stores R-squared for each repetition

# Number of folds for cross-validation
n_splits = 5

# Copy and convert ages data to integer format
Y = np.copy(input_ages)
Y = Y.astype(int)

# Lists to store indices and predictions across repetitions
test_pool_reps = []
Y_pred_pool_reps = []

# Initialize arrays to store gaps for each group across folds and repetitions
gap_high_tango = np.zeros((n_splits, reps, vectorized_high_tango.shape[1]))
gap_low_tango = np.zeros((n_splits, reps, vectorized_low_tango.shape[1]))
gap_SC1 = np.zeros((n_splits, reps, vectorized_SC1.shape[1]))
gap_SC2 = np.zeros((n_splits, reps, vectorized_SC2.shape[1]))
gap_nonvisual = np.zeros((n_splits, reps, vectorized_nonvisual.shape[1]))
gap_visual = np.zeros((n_splits, reps, vectorized_visual.shape[1]))
gap_sonata_post = np.zeros((n_splits, reps, vectorized_sonata_post.shape[1]))
gap_sonata_pre = np.zeros((n_splits, reps, vectorized_sonata_pre.shape[1]))
gap_sonata_post_active = np.zeros((n_splits, reps, vectorized_sonata_post_active.shape[1]))
gap_sonata_pre_active = np.zeros((n_splits, reps, vectorized_sonata_pre_active.shape[1]))

# Array to store feature importances for each repetition and fold
feature_importances = np.zeros((reps, n_splits, new_pairs))

# Start timing the process
init = time.time()

# CNN settings
cnn_epochs = 40
cnn_batch_size = 64
cnn_learning_rate = 1e-3
cnn_weight_decay = 1e-4

# Main loop to repeat CNN training and testing for `reps` repetitions
for k in range(reps):

    # Lists to store metrics for each fold in this repetition
    rtemp_pool = []
    error_pool = []
    rmse_pool = []
    r2_pool = []

    # 5-fold cross-validation setup
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=k)
    counter = 0  # Fold index

    # Loop over each fold
    for train, test in cv.split(Y, Y):
        seed = (k * 100) + counter

        # Separate training and test data for the fold
        Y_train = Y[train].astype(np.float32)
        Y_test = Y[test].astype(np.float32)

        X_pool_train = vectorized[:, train].T.astype(np.float32)
        X_pool_test = vectorized[:, test].T.astype(np.float32)

        aux_arrays = [
            X_pool_train, X_pool_test,
            vectorized_SC1.T.astype(np.float32), vectorized_SC2.T.astype(np.float32),
            vectorized_high_tango.T.astype(np.float32), vectorized_low_tango.T.astype(np.float32),
            vectorized_sonata_pre.T.astype(np.float32), vectorized_sonata_post.T.astype(np.float32),
            vectorized_sonata_pre_active.T.astype(np.float32), vectorized_sonata_post_active.T.astype(np.float32),
            vectorized_visual.T.astype(np.float32), vectorized_nonvisual.T.astype(np.float32),
        ]

        standardized, feat_mean, feat_std = standardize_features(X_pool_train, aux_arrays)
        (X_train_std, X_test_std, X_SC1_std, X_SC2_std, X_high_tango_std, X_low_tango_std,
         X_sonata_pre_std, X_sonata_post_std, X_sonata_pre_active_std, X_sonata_post_active_std,
         X_visual_std, X_nonvisual_std) = standardized

        model, best_val_loss = train_cnn_regressor(
            X_train_std, Y_train, seed=seed, epochs=cnn_epochs, batch_size=cnn_batch_size,
            learning_rate=cnn_learning_rate, weight_decay=cnn_weight_decay
        )

        # Predict on test data and store fold-level metrics
        Y_pred = predict_with_cnn(model, X_test_std)
        fold_r = stats.pearsonr(Y_test, Y_pred)[0]
        fold_mae = mean_absolute_error(Y_test, Y_pred)
        fold_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        fold_r2 = r2_score(Y_test, Y_pred)

        rtemp_pool.append(fold_r)
        error_pool.append(fold_mae)
        rmse_pool.append(fold_rmse)
        r2_pool.append(fold_r2)

        test_pool_reps.append(test)
        Y_pred_pool_reps.append(Y_pred)

        # Calculate regression line for training gap (age-bias correction)
        train_pred = predict_with_cnn(model, X_train_std)
        gap_train = (train_pred - Y_train)
        a, b = stats.linregress(Y_train, gap_train)[0:2]

        # Calculate gaps for each group using the trained model
        Y_pred_SC1 = predict_with_cnn(model, X_SC1_std)
        gap_SC1[counter, k, :] = (Y_pred_SC1 - ages_SC1) - (a * ages_SC1 + b)

        Y_pred_SC2 = predict_with_cnn(model, X_SC2_std)
        gap_SC2[counter, k, :] = (Y_pred_SC2 - ages_SC2) - (a * ages_SC2 + b)

        Y_pred_high_tango = predict_with_cnn(model, X_high_tango_std)
        gap_high_tango[counter, k, :] = (Y_pred_high_tango - ages_high_tango) - (a * ages_high_tango + b)

        Y_pred_low_tango = predict_with_cnn(model, X_low_tango_std)
        gap_low_tango[counter, k, :] = (Y_pred_low_tango - ages_low_tango) - (a * ages_low_tango + b)

        Y_pred_sonata_pre = predict_with_cnn(model, X_sonata_pre_std)
        gap_sonata_pre[counter, k, :] = (Y_pred_sonata_pre - ages_sonata) - (a * ages_sonata + b)

        Y_pred_sonata_post = predict_with_cnn(model, X_sonata_post_std)
        gap_sonata_post[counter, k, :] = (Y_pred_sonata_post - ages_sonata) - (a * ages_sonata + b)

        Y_pred_sonata_pre_active = predict_with_cnn(model, X_sonata_pre_active_std)
        gap_sonata_pre_active[counter, k, :] = (Y_pred_sonata_pre_active - ages_sonata_active) - (a * ages_sonata_active + b)

        Y_pred_sonata_post_active = predict_with_cnn(model, X_sonata_post_active_std)
        gap_sonata_post_active[counter, k, :] = (Y_pred_sonata_post_active - ages_sonata_active) - (a * ages_sonata_active + b)

        Y_pred_visual = predict_with_cnn(model, X_visual_std)
        gap_visual[counter, k, :] = (Y_pred_visual - ages_visual) - (a * ages_visual + b)

        Y_pred_nonvisual = predict_with_cnn(model, X_nonvisual_std)
        gap_nonvisual[counter, k, :] = (Y_pred_nonvisual - ages_nonvisual) - (a * ages_nonvisual + b)

        # Placeholder for feature importances; CNN does not expose them directly in this pipeline
        feature_importances[k, counter, :] = 0

        counter += 1

    # Save average results across folds for this repetition
    rreps[k] = np.mean(rtemp_pool)
    ereps[k] = np.mean(error_pool)
    rmse_reps[k] = np.mean(rmse_pool)
    r2_reps[k] = np.mean(r2_pool)

    print(
        f"Repetition {k}: r = {rreps[k]:.4f}, MAE = {ereps[k]:.4f}, "
        f"RMSE = {rmse_reps[k]:.4f}, R2 = {r2_reps[k]:.4f}"
    )

# Print total time taken for the loop
print(f"Total training time: {time.time() - init:.2f} seconds")
print(
    f"CNN CV summary -> mean r = {np.mean(rreps):.4f} ± {np.std(rreps):.4f}, "
    f"MAE = {np.mean(ereps):.4f} ± {np.std(ereps):.4f}, "
    f"RMSE = {np.mean(rmse_reps):.4f} ± {np.std(rmse_reps):.4f}, "
    f"R2 = {np.mean(r2_reps):.4f} ± {np.std(r2_reps):.4f}"
)

with open(os.path.join(RESULTS_DIR, "cnn_model_summary_CNN.txt"), "w", encoding="utf-8") as summary_file:
    summary_file.write(
        "CNN settings\n"
        f"epochs: {cnn_epochs}\n"
        f"batch_size: {cnn_batch_size}\n"
        f"learning_rate: {cnn_learning_rate}\n"
        f"weight_decay: {cnn_weight_decay}\n"
        f"repetitions: {reps}\n"
        f"folds: {n_splits}\n"
        f"device: {DEVICE}\n\n"
        f"Mean Pearson r: {np.mean(rreps):.6f}\n"
        f"Std Pearson r: {np.std(rreps):.6f}\n"
        f"Mean MAE: {np.mean(ereps):.6f}\n"
        f"Std MAE: {np.std(ereps):.6f}\n"
        f"Mean RMSE: {np.mean(rmse_reps):.6f}\n"
        f"Std RMSE: {np.std(rmse_reps):.6f}\n"
        f"Mean R2: {np.mean(r2_reps):.6f}\n"
        f"Std R2: {np.std(r2_reps):.6f}\n"
    )
print(f"Saved model summary: {os.path.join(RESULTS_DIR, 'cnn_model_summary_CNN.txt')}")

# Calculate mean gaps for each group across folds and repetitions
gap_high_tango = np.mean(np.mean(gap_high_tango, 0), 0)
gap_low_tango = np.mean(np.mean(gap_low_tango, 0), 0)
gap_SC1 = np.mean(np.mean(gap_SC1, 0), 0)
gap_SC2 = np.mean(np.mean(gap_SC2, 0), 0)
gap_nonvisual = np.mean(np.mean(gap_nonvisual, 0), 0)
gap_visual = np.mean(np.mean(gap_visual, 0), 0)
gap_sonata_post = np.mean(np.mean(gap_sonata_post, 0), 0)
gap_sonata_pre = np.mean(np.mean(gap_sonata_pre, 0), 0)
gap_sonata_post_active = np.mean(np.mean(gap_sonata_post_active, 0), 0)
gap_sonata_pre_active = np.mean(np.mean(gap_sonata_pre_active, 0), 0)


# For compunting the standardize brain age gaps (BAGs) across all expert groups to obtain z-scores
all_bags = np.concatenate((gap_high_tango, gap_low_tango, 
                           gap_SC1, gap_SC2, 
                           gap_visual, gap_nonvisual)) 

# Adjust gaps for each group by subtracting their pairwise mean
mean_gaming = 0.5 * (np.mean(gap_SC1) + np.mean(gap_SC2))
gap_SC1, gap_SC2 = gap_SC1 - mean_gaming, gap_SC2 - mean_gaming

mean_tango = 0.5 * (np.mean(gap_high_tango) + np.mean(gap_low_tango))
gap_high_tango, gap_low_tango = gap_high_tango - mean_tango, gap_low_tango - mean_tango

mean_visual = 0.5 * (np.mean(gap_visual) + np.mean(gap_nonvisual))
gap_visual, gap_nonvisual = gap_visual - mean_visual, gap_nonvisual - mean_visual

mean_sonata = 0.5 * (np.mean(gap_sonata_post) + np.mean(gap_sonata_pre))
gap_sonata_post, gap_sonata_pre = gap_sonata_post - mean_sonata, gap_sonata_pre - mean_sonata

mean_sonata_active = 0.5 * (np.mean(gap_sonata_post_active) + np.mean(gap_sonata_pre_active))
gap_sonata_post_active, gap_sonata_pre_active = gap_sonata_post_active - mean_sonata_active, gap_sonata_pre_active - mean_sonata_active


#%%

# Initialize an array to store predictions across all repetitions
# Each row corresponds to a participant, and each column to a repetition
Y_pred_all = np.zeros((np.sum(input_ages > 0), reps))

# Populate Y_pred_all with predictions from each repetition
for k in range(0, reps):
    Y_pred_all[test_pool_reps[k], k] = Y_pred_pool_reps[k]

# Compute the mean predicted age for each participant across all repetitions
Y_pred_all = np.sum(Y_pred_all, 1) / np.sum(Y_pred_all > 0, 1)

# Plotting the combined model (observed vs predicted age)
plt.figure(1, figsize=(5, 4.5))
plt.clf()  # Clear the figure

# Scatter plot of actual vs predicted ages
plt.plot(Y, Y_pred_all, 'bo')  # 'bo' for blue circles

# Perform linear regression on actual vs predicted ages
a, b, r = stats.linregress(Y, Y_pred_all)[0:3]
lines = b + a * Y  # Calculate regression line

# Plot the regression line on the scatter plot
plt.plot(Y, lines, color='crimson', lw=1.5, ls='dashed')  # 'dashed' for dashed line style

# Set plot labels and title with Pearson correlation value
plt.xlabel('Chronological age (years)')
plt.ylabel('Predicted age (years)')
plt.title(
    "RF: r = %.3f | MAE = %.2f | RMSE = %.2f | R2 = %.3f"
    % (np.mean(rreps), np.mean(ereps), np.mean(rmse_reps), np.mean(r2_reps))
)

# Set x and y axis limits for the plot
plt.xlim(-10, 110)
plt.ylim(-10, 110)
save_figure('figure_1_brain_age_model.png')

#%%


# Create a figure with 6 subplots arranged in a single row
plt.figure(2, figsize=(11, 7.5))
plt.clf()  # Clear the figure

# Tango group
ax = plt.subplot(2, 3, 1)
violin_plot(ax, [gap_high_tango, gap_low_tango], ['skyblue', 'salmon'], 0.8, 20, 20)
print_stats("Tango dancers", gap_high_tango, gap_low_tango)
plt.ylabel('BAG (years)', fontsize=18)
plt.xlabel('Groups', fontsize=18)
plt.title('Tango dancers', fontsize=18)
plt.ylim(-35 - 4.5, 55)
plt.xticks([0, 1], ['Experts', 'Non-Experts'], fontsize=15)
plt.yticks([-35, -5, 25, 55], fontsize=15)

# Musicians group
ax = plt.subplot(2, 3, 2)
plt.ylabel('BAG (years)', fontsize=18)
plt.xlabel('Groups', fontsize=18)
plt.title('Musicians', fontsize=18)
plt.ylim(-27 - 3, 33)
plt.xticks([0, 1], ['Experts', 'Non-Experts'], fontsize=15)
plt.yticks([-27, -7, 13, 33], fontsize=15)

# Visual artists group
ax = plt.subplot(2, 3, 3)
violin_plot(ax, [gap_visual, gap_nonvisual], ['skyblue', 'salmon'], 0.8, 20, 20)
print_stats("Visual artists", gap_visual, gap_nonvisual)
plt.ylabel('BAG (years)', fontsize=18)
plt.xlabel('Groups', fontsize=18)
plt.title('Visual artists', fontsize=18)
plt.ylim(-16 - 1.8, 20)
plt.xticks([0, 1], ['Experts', 'Non-Experts'], fontsize=15)
plt.yticks([-16, -4, 8, 20], fontsize=15)

# Gaming group
ax = plt.subplot(2, 3, 4)
violin_plot(ax, [gap_SC1, gap_SC2], ['skyblue', 'salmon'], 0.8, 20, 20)
print_stats("Gaming", gap_SC1, gap_SC2)
plt.ylabel('BAG (years)', fontsize=18)
plt.xlabel('Groups', fontsize=18)
plt.title('Gaming', fontsize=18)
plt.ylim(-31 - 2.85, 27)
plt.xticks([0, 1], ['Experts', 'Non-Experts'], fontsize=15)
plt.yticks([-30, -11, 8, 27], fontsize=15)

# Learners group (Sonata pre-post)
ax = plt.subplot(2, 3, 5)
violin_plot(ax, [gap_sonata_post, gap_sonata_pre], ['skyblue', 'salmon'], 0.8, 20, 20)
print_stats("Learners", gap_sonata_post, gap_sonata_pre, paired=True)
plt.ylabel('BAG (years)', fontsize=18)
plt.xlabel('Conditions', fontsize=18)
plt.title('Learners', fontsize=18)
plt.ylim(-15 - 2.25, 30)
plt.xticks([0, 1], ['Post', 'Pre'], fontsize=15)
plt.yticks([-15, 0, 15, 30], fontsize=15)

# Learners control group
ax = plt.subplot(2, 3, 6)
violin_plot(ax, [gap_sonata_post_active, gap_sonata_pre_active], ['skyblue', 'salmon'], 0.8, 20, 20)
print_stats("Active control", gap_sonata_post_active, gap_sonata_pre_active, paired=True)
plt.ylabel('BAG (years)', fontsize=18)
plt.xlabel('Conditions', fontsize=18)
plt.title('Active control', fontsize=18)
plt.ylim(-15 - 2.25, 30)
plt.xticks([0, 1], ['Post', 'Pre'], fontsize=15)
plt.yticks([-15, 0, 15, 30], fontsize=15)


# Adjust layout
plt.tight_layout()
save_figure('figure_2_bag_group_comparisons.png')

# Store uncorrected p-values and labels
pvals = []
labels = []

# Tango group
...
t_stat, p_val = stats.ttest_ind(gap_high_tango, gap_low_tango)
d = cohen_d(gap_high_tango, gap_low_tango)
delta = np.mean(gap_high_tango) - np.mean(gap_low_tango)
pvals.append(p_val)
labels.append("Tango dancers")

# Visual artists
...
t_stat, p_val = stats.ttest_ind(gap_visual, gap_nonvisual)
...
pvals.append(p_val)
labels.append("Visual artists")

# Gaming
...
t_stat, p_val = stats.ttest_ind(gap_SC1, gap_SC2)
...
pvals.append(p_val)
labels.append("Gaming")

# Learners (paired)
...
t_stat, p_val = stats.ttest_rel(gap_sonata_post, gap_sonata_pre)
...
pvals.append(p_val)
labels.append("Learners")

# Learners (control)
...
t_stat, p_val = stats.ttest_rel(gap_sonata_post_active, gap_sonata_pre_active)
...
pvals.append(p_val)
labels.append("Active control")

# After all groups are tested: FDR correction
_, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

# Print corrected p-values
print("\nFDR-corrected p-values:")
for label, p_corr in zip(labels, pvals_corrected):
    print(f"{label}: corrected p = {p_corr:.4f}")



#%%

# Initialize a new figure with a 2x2 grid layout for subplots
plt.figure(3, figsize=(9, 8))
plt.clf()  # Clear the figure

# Subplot 1: General Expertise vs. BAGs
ax = plt.subplot(2, 2, 1)

# Combine hours and gaps for Tango group
hours_tango = np.append(hours_high_tango, hours_low_tango)
gap_tango = np.append(gap_high_tango, gap_low_tango)

# Z-score normalization for expertise-related measures
zh_tango = stats.zscore(hours_tango[hours_tango > 0])
zh_games = stats.zscore(playing_time[playing_time > 0])
zh_visual = stats.zscore(experience_visual)

# Z-score normalization for BAGs
zg_tango = stats.zscore(gap_tango)[hours_tango > 0]
zg_games = stats.zscore(gap_SC1)[playing_time > 0]
zg_visual = stats.zscore(gap_visual)

# Store p-values
pvals = []

# Combine data
xt = np.concatenate((zh_tango, zh_visual, zh_games))
yt = np.concatenate((zg_tango, zg_visual, zg_games))

# Plot
plt.scatter(xt, yt, c='darkblue', s=100, alpha=0.5)
a, b, r, p = stats.linregress(xt, yt)[0:4]
pvals.append(p)
plt.plot(xt, xt * a + b, ls='solid', color='black', lw=3)

# Labels
plt.xlabel('Z-scored expertise', fontsize=18)
plt.ylabel('Z-scored BAGs', fontsize=18)
plt.title('General Expertise', fontsize=18)
plt.xticks([-3.5, 0, 3.5], fontsize=15)
plt.yticks([-3.5, 0, 3.5], fontsize=15)
plt.ylim(-3.5, 3.5)
plt.xlim(-3.5, 3.5)

# Print correlation
print_corr("General Expertise", xt, yt)

# Subplot 3: APM Pre vs. Post
ax = plt.subplot(2, 2, 3)

APM_2 = np.load('Learning/APM_post.npy')
APM_1 = np.load('Learning/APM_pre.npy')
APM_delta = APM_2 - APM_1
APM_idx = np.isnan(APM_delta) == False

delta_sonata = gap_sonata_post - gap_sonata_pre

violin_plot(ax, [APM_2[APM_idx], APM_1[APM_idx]], ['skyblue', 'salmon'], 0.8, 20, 20)

# Print stats
print_paired("APM Pre vs Post", APM_2[APM_idx], APM_1[APM_idx])

plt.ylabel('APM', fontsize=18)
plt.xlabel('Conditions', fontsize=18)
plt.title('Actions per minute', fontsize=18)
plt.ylim(12 - 2.1, 54)
plt.xticks([0, 1], ['Post', 'Pre'], fontsize=15)
plt.yticks([12., 26., 40, 54], fontsize=15)

# Subplot 4: ΔAPM vs. ΔBAGs
ax = plt.subplot(2, 2, 4)

xt, yt = APM_delta[APM_idx], delta_sonata[APM_idx]
plt.scatter(xt, yt, c='darkblue', s=100, alpha=0.5)
a, b, r, p = stats.linregress(xt, yt)[0:4]
pvals.append(p)
plt.plot(xt, xt * a + b, ls='solid', color='black', lw=3)

plt.xlabel(r'$\Delta$ APM', fontsize=18)
plt.ylabel(r'$\Delta$ BAGs (years)', fontsize=18)
plt.title('Post training performance', fontsize=18)
plt.xticks([-2., 5.5, 13.], fontsize=15)
plt.yticks([-15, 0, 15], fontsize=15)
plt.ylim(-15 - 1.5, 15)
plt.xlim(-2, 12)

# Print correlation
print_corr("ΔAPM vs ΔBAGs", xt, yt)

# Final layout adjustment
plt.tight_layout()
save_figure('figure_3_expertise_and_performance.png')

# Adjust p-values with FDR correction
_, p_adj, _, _ = multipletests(pvals, method='fdr_bh')
print(f"Adjusted p-values (FDR): {np.round(p_adj, 4)}")

#%%
# Define the range of thresholds for calculating metrics
thresholds = np.linspace(0.02, 0.1, num=9)  # Range from 0.02 to 0.1 with 9 steps

# Helper function to calculate the Area Under the Curve (AUC) for each metric
def calculate_auc(values, thresholds):
    return np.trapz(values, thresholds)  # Uses trapezoidal rule for AUC calculation

# Function to calculate Global Efficiency (GE) and Local Efficiency (LE) for a range of thresholds
def calculate_metrics_across_thresholds(FCs, thresholds):
    GE = np.zeros((len(thresholds), FCs.shape[2]))  # Array to store GE for each threshold and matrix
    LE = np.zeros((len(thresholds), FCs.shape[2]))  # Array to store LE for each threshold and matrix
    
    # Loop over thresholds and matrices to compute binary efficiency metrics
    for j, th in enumerate(thresholds):
        for i in range(FCs.shape[2]):
            # Threshold functional connectivity matrix and binarize it
            FC_th = bct.threshold_proportional(FCs[:, :, i], th)
            FC_th[FC_th > 0] = 1
            
            # Calculate and store global and local efficiency
            GE[j, i] = bct.efficiency_bin(FC_th)
            LE[j, i] = np.mean(bct.efficiency_bin(FC_th, local=True))
    
    # Calculate AUC for GE and LE across thresholds
    GE_auc = np.mean(GE, 0)
    LE_auc = np.mean(LE, 0)
    
    return GE_auc, LE_auc  # Returns averaged AUC values for GE and LE

# Function to calculate and adjust GE and LE metrics for two groups
def calculate_and_correct_metrics(FCs_group1, FCs_group2, thresholds):
    # Calculate GE and LE for both groups
    GE_group1_auc, LE_group1_auc = calculate_metrics_across_thresholds(FCs_group1, thresholds)
    GE_group2_auc, LE_group2_auc = calculate_metrics_across_thresholds(FCs_group2, thresholds)
    
    # Combine all GE and LE values for both groups
    all_GE = np.concatenate((GE_group1_auc, GE_group2_auc))
    all_LE = np.concatenate((LE_group1_auc, LE_group2_auc))
    
    # Split the combined metrics back into individual groups
    GE_group1 = all_GE[:FCs_group1.shape[2]]
    GE_group2 = all_GE[FCs_group1.shape[2]:]
    LE_group1 = all_LE[:FCs_group1.shape[2]]
    LE_group2 = all_LE[FCs_group1.shape[2]:]
    
    return GE_group1, GE_group2, LE_group1, LE_group2  # Return separated metrics

# Calculate and adjust metrics for each pair of groups using the defined functions
GE_high_tango, GE_low_tango, LE_high_tango, LE_low_tango = calculate_and_correct_metrics(FCs_high_tango, FCs_low_tango, thresholds)
GE_SC1, GE_SC2, LE_SC1, LE_SC2 = calculate_and_correct_metrics(FCs_SC1, FCs_SC2, thresholds)
GE_nonvisual, GE_visual, LE_nonvisual, LE_visual = calculate_and_correct_metrics(FCs_nonvisual, FCs_visual, thresholds)
GE_sonata_post, GE_sonata_pre, LE_sonata_post, LE_sonata_pre = calculate_and_correct_metrics(FCs_sonata_post, FCs_sonata_pre, thresholds)


#%%

# Load pre-computed global coupling values
Gs_experts = np.load('Global_coupling/Gs_experts.npy')
Gs_training = np.load('Global_coupling/Gs_training.npy')

# Z-score normalization for BAGs
z_bags = stats.zscore(all_bags)

# Combine GE and LE metrics across expert groups
all_GE = np.concatenate((GE_high_tango, GE_low_tango, GE_SC1, GE_SC2, GE_nonvisual, GE_visual))
all_LE = np.concatenate((LE_high_tango, LE_low_tango, LE_SC1, LE_SC2, LE_nonvisual, LE_visual))

ps = []
labels = []

plt.figure(4, figsize=(12, 7.25))
plt.clf()

# Integration
plt.subplot(2, 3, 1)
fx = all_LE > 0.1
xt, yt = all_GE[fx], z_bags[fx]
plt.scatter(xt, yt, c='darkblue', s=100, alpha=0.5)
a, b, r, p = stats.linregress(xt, yt)[0:4]
plt.plot(xt, xt * a + b, ls='solid', color='black', lw=3)
plt.xlabel('Global Efficiency', fontsize=18)
plt.ylabel('Z-scored BAGs', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Integration', fontsize=18)
plt.ylim(-3.85, 3.5)
plt.xlim(0.09, 0.33)
plt.xticks([0.09, 0.17, 0.25, 0.33])
plt.yticks([-3.5, 0, 3.5])
ps.append(print_corr("Integration", xt, yt))
labels.append("Integration")

# Segregation
plt.subplot(2, 3, 2)
xt, yt = all_LE[fx], z_bags[fx]
plt.scatter(xt, yt, c='darkblue', s=100, alpha=0.5)
a, b, r, p = stats.linregress(xt, yt)[0:4]
plt.plot(xt, xt * a + b, ls='solid', color='black', lw=3)
plt.xlabel('Local Efficiency', fontsize=18)
plt.title('Segregation', fontsize=18)
plt.xticks(fontsize=15)
plt.ylim(-3.85, 3.5)
plt.xlim(0.32, 0.62)
plt.xticks([0.32, 0.42, 0.52, 0.62])
plt.yticks([])
ps.append(print_corr("Segregation", xt, yt))
labels.append("Segregation")

# Modeling
plt.subplot(2, 3, 3)
xt, yt = Gs_experts[fx], z_bags[fx]
plt.scatter(xt, yt, c='darkblue', s=100, alpha=0.5)
a, b, r, p = stats.linregress(xt, yt)[0:4]
plt.plot(xt, xt * a + b, ls='solid', color='black', lw=3)
plt.xlabel('Log10 Global Coupling', fontsize=18)
plt.title('Modeling', fontsize=18)
plt.xticks(fontsize=15)
plt.ylim(-3.85, 3.5)
plt.xlim(-1.41, 0.72)
plt.xticks([-1.41, -0.7, 0.01, 0.72])
plt.yticks([])
ps.append(print_corr("Modeling", xt, yt))
labels.append("Modeling")

# Δ Global Efficiency
plt.subplot(2, 3, 4)
xt, yt = GE_sonata_post - GE_sonata_pre, gap_sonata_post - gap_sonata_pre
plt.scatter(xt, yt, c='darkblue', s=100, alpha=0.5)
a, b, r, p = stats.linregress(xt, yt)[0:4]
plt.plot(xt, xt * a + b, ls='solid', color='black', lw=3)
plt.xlabel(r'$\Delta$ Global Efficiency', fontsize=18)
plt.ylabel(r'$\Delta$ BAGs (years)', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(-16.35, 12)
plt.xlim(-0.035, 0.034)
plt.xticks([-0.035, -0.012, 0.011, 0.034])
plt.yticks([-15, -6, 3, 12])
ps.append(print_corr("Δ Global Efficiency", xt, yt))
labels.append("Δ Global Efficiency")

# Δ Local Efficiency
plt.subplot(2, 3, 5)
xt, yt = LE_sonata_post - LE_sonata_pre, gap_sonata_post - gap_sonata_pre
plt.scatter(xt, yt, c='darkblue', s=100, alpha=0.5)
a, b, r, p = stats.linregress(xt, yt)[0:4]
plt.plot(xt, xt * a + b, ls='solid', color='black', lw=3)
plt.xlabel(r'$\Delta$ Local Efficiency', fontsize=18)
plt.xticks(fontsize=15)
plt.ylim(-16.35, 12)
plt.xlim(-0.06, 0.069)
plt.xticks([-0.06, -0.017, 0.026, 0.069])
plt.yticks([])
ps.append(print_corr("Δ Local Efficiency", xt, yt))
labels.append("Δ Local Efficiency")

# Δ Global Coupling (Training)
plt.subplot(2, 3, 6)
xt, yt = Gs_training, gap_sonata_post - gap_sonata_pre
plt.scatter(xt, yt, c='darkblue', s=100, alpha=0.5)
a, b, r, p = stats.linregress(xt, yt)[0:4]
plt.plot(xt, xt * a + b, ls='solid', color='black', lw=3)
plt.xlabel(r'$\Delta$ Log10 Global Coupling', fontsize=18)
plt.xticks(fontsize=15)
plt.ylim(-16.35, 12)
plt.xlim(-0.6, 0.6)
plt.xticks([-0.6, -0.2, 0.2, 0.6])
plt.yticks([])
ps.append(print_corr("Δ Global Coupling", xt, yt))
labels.append("Δ Global Coupling")

# Apply Benjamini-Hochberg correction
_, corrected_p_values, _, _ = multipletests(ps, alpha=0.05, method='fdr_bh')
print_corrected_pvals(labels, corrected_p_values)

plt.tight_layout()
save_figure('figure_4_efficiency_and_coupling.png')


#%%

# Load maps for correlations with age vulnerability and connectivity effect sizes
corr_vec = np.load("neurosynth_spin_test/Ds_corrs.npy") * -1  # Age vulnerability map

#Connectivity effect size for experts
Ds_music = np.load("neurosynth_spin_test/Ds_music.npy") 
Ds_tango = np.load("neurosynth_spin_test/Ds_tango.npy") 
Ds_visual = np.load("neurosynth_spin_test/Ds_visual.npy")
Ds_gaming = np.load("neurosynth_spin_test/Ds_gaming.npy")

#Averaged map across domains 
Ds_experts = (Ds_tango + Ds_music + Ds_visual + Ds_gaming) / 4

#Pre/post-learning design
Ds_training = np.load("neurosynth_spin_test/Ds_training.npy") # Connectivity effect size for learners

# Set up data and names for creating brain maps
maps = [Ds_experts, Ds_training, corr_vec]
names = ['Experts', 'Training', 'Age vulnerability']  # Names for each map

# Load GIFTI label file (left hemisphere)
lh_labels_gii = nb.load('neurosynth_spin_test/AAL.32k.L.label.gii')
lh_labels = lh_labels_gii.darrays[0].data.astype(int)  # Get label index per vertex

# Do the same for the right hemisphere
rh_labels_gii = nb.load('neurosynth_spin_test/AAL.32k.R.label.gii')
rh_labels = rh_labels_gii.darrays[0].data.astype(int)

plt.figure(20)
plt.clf()
for sx in range(0, len(maps)):

    Ds_left = maps[sx][::2]
    Ds_left = np.insert(Ds_left, 18, 0) 
    Ds_left = np.insert(Ds_left, 20, 0)
    Ds_right = maps[sx][1::2]
    Ds_right = np.insert(Ds_right, 18, 0)
    Ds_right = np.insert(Ds_right, 20, 0)

    #Comment about the 0s for brain areas 17 (L,R) and 20 (L,R)
    #These brain areas are the amygdala and hippocampus, that were not used in our study
    #We needed to assign values to these brain areas just for plotting, although we did not use them for any of the analyses
    
    # Map each vertex to its corresponding value
    lh_vertex_data = np.zeros_like(lh_labels, dtype=float)
    for i in range(41):
        lh_vertex_data[lh_labels == i+1] = Ds_left[i]    
    
    rh_vertex_data = np.zeros_like(rh_labels, dtype=float)
    for i in range(41):
        rh_vertex_data[rh_labels == i+1] = Ds_right[i]
    
    # plt.title(names[sx])
    
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']
    p = Plot(lh, rh, size = (1000,800))
    
    # add schaefer parcellation (no color bar needed)
    vmin = -np.max(np.abs(np.append(lh_vertex_data,rh_vertex_data)))
    vmax = -vmin
    p.add_layer({'left': lh_vertex_data, 'right': rh_vertex_data}, 
                cmap = 'RdBu_r', cbar = True, color_range = (vmin, vmax))
    
    
    fig = p.build()
    fig.text(y = 0.9, x = 0.25, s = names[sx], fontsize = 40)
    save_figure(f"figure_20_{_sanitize_filename(names[sx])}.png", fig=fig)
    fig.show()
plt.close(20)    

#%%    
# Set adjustable font sizes for plot elements
tick_label_fontsize = 16  # Font size for axis tick labels
label_fontsize = 19       # Font size for axis labels
title_fontsize = 19       # Font size for plot titles
text_fontsize = 16        # Font size for text annotations

# Plot for Ds_experts (Experts' data)
plt.figure(6)  # Initialize figure 6
plt.clf()      # Clear the current figure


# Store raw p-values and their labels
p_values = []
labels = []

# === Plot for Experts ===
plt.figure(6)
plt.clf()

# Linear regression
a, b, r, p = stats.linregress(corr_vec, Ds_experts)[0:4]
plt.scatter(corr_vec, Ds_experts, s=120, alpha=0.5, color='blue')

# Distance matrix
aal_coords = pd.read_csv("neurosynth_spin_test/AAL_coordinates.txt", sep = ' ', header = None) #x,y,z coordinates
aal_coords = aal_coords.iloc[0:90, 0:3].to_numpy()
dist_matrix = squareform(pdist(aal_coords))[0:90, 0:90]
dist_matrix = np.delete(dist_matrix, np.array([36,37,40,41,70,71,72,73,74,75,76,76,77]), axis=0)
dist_matrix = np.delete(dist_matrix, np.array([36,37,40,41,70,71,72,73,74,75,76,76,77]), axis=1)

# Surrogate testing
base_D = Base(x=Ds_experts, D=dist_matrix)
surr_number = 10000
surrogates_D = base_D(n=surr_number)

true_corr = stats.pearsonr(corr_vec, Ds_experts)[0]
surr_corr = np.array([stats.pearsonr(surrogates_D[i, :], corr_vec)[0] for i in range(surr_number)])
p_value = np.mean(surr_corr < true_corr) if true_corr < 0 else np.mean(surr_corr > true_corr)

# Print result with label
print(f"Experts, r = {r:.4f}, surrogate p = {p_value:.4f}")
p_values.append(p_value)
labels.append("Experts")

# Plot regression line
regression_line = a * corr_vec + b
plt.plot(corr_vec, regression_line, color='black', linestyle='-', linewidth=1.5)
plt.text(0.25, 0.95, f'$r = {r:.3f}$',
         transform=plt.gca().transAxes, fontsize=text_fontsize,
         verticalalignment='top', horizontalalignment='right')
plt.xlabel('Negative of correlations', fontsize=label_fontsize)
plt.ylabel('Effect size', fontsize=label_fontsize)
plt.title('Experts', fontsize=title_fontsize)
set_limits_and_ticks(plt.gca(), -0.06, 0.3, -0.08, 0.46)
plt.xticks(fontsize=tick_label_fontsize)
plt.yticks(fontsize=tick_label_fontsize)
plt.tight_layout()
save_figure('figure_6_experts_spatial_correlation.png')
plt.show()

# === Plot for Training ===
plt.figure(7)
plt.clf()

# Linear regression
a, b, r, p = stats.linregress(corr_vec, Ds_training)[0:4]
plt.scatter(corr_vec, Ds_training, s=120, alpha=0.5, color='red')

# Surrogate testing
base_D = Base(x=Ds_training, D=dist_matrix)
surrogates_D = base_D(n=surr_number)

true_corr = stats.pearsonr(corr_vec, Ds_training)[0]
surr_corr = np.array([stats.pearsonr(surrogates_D[i, :], corr_vec)[0] for i in range(surr_number)])
p_value = np.mean(surr_corr < true_corr) if true_corr < 0 else np.mean(surr_corr > true_corr)

# Print result with label
print(f"Training, r = {r:.4f}, surrogate p = {p_value:.4f}")
p_values.append(p_value)
labels.append("Training")

# Plot regression line
regression_line = a * corr_vec + b
plt.plot(corr_vec, regression_line, color='black', linestyle='-', linewidth=1.5)
plt.text(0.25, 0.95, f'$r = {r:.3f}$',
         transform=plt.gca().transAxes, fontsize=text_fontsize,
         verticalalignment='top', horizontalalignment='right')
plt.xlabel('Negative of correlations', fontsize=label_fontsize)
plt.ylabel('Effect size', fontsize=label_fontsize)
plt.title('Training', fontsize=title_fontsize)
set_limits_and_ticks(plt.gca(), -0.06, 0.3, 0.1, 0.55)
plt.xticks(fontsize=tick_label_fontsize)
plt.yticks(fontsize=tick_label_fontsize)
plt.tight_layout()
save_figure('figure_7_training_spatial_correlation.png')
plt.show()

# === FDR Correction ===
_, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

print("\nFDR-corrected p-values:")
for label, corrected_p in zip(labels, pvals_corrected):
    print(f"{label}: corrected p = {corrected_p:.4f}")

#%%
#surrogate data
base_experts = Base(x = Ds_experts, D = dist_matrix)
surr_number = 10000
surrogates_experts = base_experts(n = surr_number)

base_training = Base(x = Ds_training, D = dist_matrix)
surr_number = 10000
surrogates_training = base_training(n = surr_number)

# Load parcellated data and cognitive terms
parcellated_data = np.load('neurosynth_spin_test/parcellated_data.npy')
cognitive_terms = np.load('neurosynth_spin_test/cognitive_terms.npy')

# Prepare neuro maps by selecting the first 90 regions and removing specific regions
neuro_maps = np.copy(parcellated_data)[:, 0:90]
neuro_maps = np.delete(neuro_maps, np.array([36, 37, 40, 41, 70, 71, 72, 73, 74, 75, 76, 77]), axis=1)
maps = cognitive_terms
maps = [label.replace('_', ' ') for label in maps]  # Remove underscores from labels for readability

# Initialize arrays to hold correlation coefficients and p-values for experts and training
corrs_experts = np.zeros(len(maps))
pvals_experts = np.zeros(len(maps))
corrs_training = np.zeros(len(maps))
pvals_training = np.zeros(len(maps))

# Calculate Pearson correlations and p-values between connectivity effect sizes and each neuro map
for i in range(len(maps)):
    
    #true correlation
    true_corr = stats.pearsonr(neuro_maps[i, :], Ds_experts)[0]
    #surrogate correlations
    surr_corr = np.array([stats.pearsonr(surrogates_experts[x,:], neuro_maps[i, :])[0] for x in range(0,surr_number)])

    #computing p_value
    if true_corr < 0:
        p_value = np.mean(surr_corr < true_corr)
    if true_corr >= 0:
        p_value = np.mean(surr_corr > true_corr)
    
    corrs_experts[i], pvals_experts[i] = stats.pearsonr(Ds_experts, neuro_maps[i, :])[0], p_value
   
    #true correlation
    true_corr = stats.pearsonr(neuro_maps[i, :], Ds_training)[0]
    #surrogate correlations
    surr_corr = np.array([stats.pearsonr(surrogates_training[x,:], neuro_maps[i, :])[0] for x in range(0,surr_number)])

    #computing p_value
    if true_corr < 0:
        p_value = np.mean(surr_corr < true_corr)
    if true_corr >= 0:
        p_value = np.mean(surr_corr > true_corr)   
   
    corrs_training[i], pvals_training[i] = stats.pearsonr(Ds_training, neuro_maps[i, :])
    
    print(i)

# Convert correlations to absolute values for ranking purposes
corrs_experts = np.abs(corrs_experts)
corrs_training = np.abs(corrs_training)

# Apply Benjamini-Hochberg correction for multiple comparisons
_, pvals_experts_corrected, _, _ = multipletests(pvals_experts, alpha=0.05, method='fdr_bh')
_, pvals_training_corrected, _, _ = multipletests(pvals_training, alpha=0.05, method='fdr_bh')

# Identify the indices of the top 10 correlations for each group
best_10_experts = np.argsort(corrs_experts)[::-1][:10]
best_10_training = np.argsort(corrs_training)[::-1][:10]

# Function to set border thickness based on p-value significance level
def border_thickness(p):
    if p < 0.001:
        return 6  # Highest significance
    elif p < 0.01:
        return 3   # Moderate significance
    elif p < 0.05:
        return 1  # Least significance
    else:
        return 0  # No border

# Function to clean label by removing underscores (not strictly necessary here)
def clean_label(label):
    return label.replace('_', ' ')

def plot_correlations(indices, corrs, pvals, title, color, ylabel='Cognitive Terms', xlim=(0.2, 0.5),
                      fig_number = 1):
    plt.figure(fig_number, figsize=(8.5, 7))
    plt.clf()
    
    # Scale dot sizes based on correlation values
    sizes = ((corrs[indices] - 0.3) / (np.max(corrs[indices]) - 0.3) * 0.3 + 0) * 3000
    edge_widths = [border_thickness(pvals[i]) for i in indices]
    
    for i, (corr, size, edge_width) in enumerate(zip(corrs[indices], sizes, edge_widths)):
        plt.scatter(0.5, i, s=size, color=color, alpha=1,
                    edgecolors='black' if edge_width > 0 else 'none',
                    linewidth=edge_width)
    
    # Y-axis labels
    plt.yticks(range(10), [clean_label(maps[i]) for i in indices], fontsize=16)
    plt.ylabel(ylabel, fontsize=18)
    plt.gca().invert_yaxis()
    
    # Aesthetic cleanup
    for spine in ['top', 'right', 'left', 'bottom']:
        plt.gca().spines[spine].set_visible(False)
    plt.xticks([])
    plt.xlim(0.4, 0.6)
    plt.title(title, fontsize=18)

    # Legend for significance levels including p > 0.05 (no border)
    legend_sizes = [200] * 4
    legend_colors = [color] * 4
    legend_edges = ['none', 'black', 'black', 'black']
    legend_linewidths = [0, 1, 3, 6]
    legend_labels = ['p > 0.05', 'p < 0.05', 'p < 0.01', 'p < 0.001']
    
    # Create dummy scatter points for legend
    handles = [
        plt.scatter([], [], s=legend_sizes[i], color=legend_colors[i],
                    edgecolors=legend_edges[i], linewidth=legend_linewidths[i],
                    label=legend_labels[i])
        for i in range(4)
    ]
    
    plt.legend(
        handles=handles,
        title="Significance",
        loc='lower right',
        fontsize=12,
        title_fontsize=13,
        handletextpad=1,
        labelspacing=1.2  # Adds spacing between entries
    )

    plt.tight_layout()
    save_figure(f"figure_{fig_number}_{_sanitize_filename(title)}.png")
    plt.show()

# Plot top 10 correlations for experts
plot_correlations(best_10_experts, corrs_experts, pvals_experts_corrected, 
                  'Top 10 Correlations for Experts', 'skyblue', fig_number = 9)

# Plot top 10 correlations for training
plot_correlations(best_10_training, corrs_training, pvals_training_corrected, 
                  'Top 10 Correlations for Training', 'salmon', fig_number = 10)

# Create word clouds with specific colors for experts and training

# Define color for word cloud for experts
def color_words_experts(word, font_size, position, orientation, random_state=None, **kwargs):
    return "orange"

# Create word cloud for experts
wordcloud_experts = WordCloud(width=400, height=800, background_color='white',
                              max_font_size=100, min_font_size=10,
                              color_func=color_words_experts).generate_from_frequencies(
    {maps[i]: np.abs(corrs_experts[i]) for i in range(len(maps))})

plt.figure(11, figsize=(9, 7.5))
plt.clf()
plt.imshow(wordcloud_experts, interpolation='bilinear')
plt.axis('off')
save_figure('figure_11_wordcloud_experts.png')
plt.show()

# Define color for word cloud for training
def color_words_training(word, font_size, position, orientation, random_state=None, **kwargs):
    return "SeaGreen"

# Create word cloud for training
wordcloud_training = WordCloud(width=400, height=800, background_color='white',
                               max_font_size=100, min_font_size=10,
                               color_func=color_words_training).generate_from_frequencies(
    {maps[i]: corrs_training[i] for i in range(len(maps))})

plt.figure(12, figsize=(9, 7.5))
plt.clf()
plt.imshow(wordcloud_training, interpolation='bilinear')
plt.axis('off')
save_figure('figure_12_wordcloud_training.png')
plt.show()


#%%






log_file.flush()

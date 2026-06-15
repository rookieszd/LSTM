"""
PINN + SEGA-TCN-Att-LSTM recursive multi-step creep prediction model.
Uses rollout training with delta-Jcreep targets and physics-informed losses.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import random
import warnings
import json
import os

warnings.filterwarnings('ignore')

# =========================================================
# Reproducibility settings
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark =True
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Training switch:
# False = run SEGA + train model, then save MODEL_PATH and BEST_PARAMS_PATH.
# True  = skip SEGA/training and directly load the saved model/params.
SKIP_TRAINING = False
MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_sega_tcn_att_lstm_pinn_delta_rollout.pth')
BEST_PARAMS_PATH = os.path.join(SCRIPT_DIR, 'best_sega_params_delta_rollout.json')
# =========================================================
# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

TIME_STEP = 7
ROLLOUT_STEPS = 3
print(f'Recursive rollout training steps: {ROLLOUT_STEPS}')
BATCH_SIZE = 32
HIDDEN_SIZE = 96
NUM_LAYERS = 1
DROPOUT = 0.30
LR = 0.0005
EPOCHS = 80
PATIENCE = 12
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

# Self-adaptive Evolutionary Genetic Algorithm settings.
# These defaults are intentionally small because every individual trains a model.
SEGA_POP_SIZE = 8
SEGA_GENERATIONS = 4
SEGA_FITNESS_EPOCHS = 3
SEGA_ELITE_SIZE = 2

# Physics-loss weights
LAMBDA_PHYS_MONO = 0.1
LAMBDA_PHYS_SMOOTH = 0.01

# =========================================================
# Feature definitions
# =========================================================
base_dynamic_cols = ['dt', 'Jcreep']
dynamic_cols = ['dt', 'log_dt', 'Jcreep', 'dJ_dt']
static_cols = ['cem', 'w/c', 'a/c', 'c', 'fc28', 'E28', 'V/S', 't', 'T', 'RH']
target_col = 'Jcreep'

print(f'Dynamic features ({len(dynamic_cols)}): {dynamic_cols}')
print(f'Static features ({len(static_cols)}): {static_cols}')
print(f'Total input features: {len(dynamic_cols) + len(static_cols)}')

# =========================================================
# Read and preprocess data
# =========================================================
df = pd.read_csv(r'C:\Users\91028\Desktop\while file\shrinkage_creep_data\CREEP.csv')
df = df.dropna(axis=1, how='all')
print(f'Raw data rows: {len(df)}')

numeric_cols = static_cols + base_dynamic_cols
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

min_j = df[target_col].min()
if min_j < 0:
    shift = -min_j + 1
    df[target_col] += shift
    print(f'Shifted Jcreep by {shift}')
else:
    shift = 0

df = df.sort_values(['File', 'dt']).reset_index(drop=True)
df['log_dt'] = np.log1p(df['dt'])
dt_diff = df.groupby('File')['dt'].diff()
j_diff = df.groupby('File')[target_col].diff()
df['dJ_dt'] = (j_diff / dt_diff.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
df['dJ_dt'] = df.groupby('File')['dJ_dt'].transform(lambda s: s.bfill().ffill())
df['dJ_dt'] = df['dJ_dt'].fillna(0)

all_used_cols = dynamic_cols + static_cols
valid_idx = df[all_used_cols].notna().all(axis=1)
df = df[valid_idx].reset_index(drop=True)
print(f'Valid data rows: {len(df)}')

# =========================================================
# Group data by specimen
groups = df.groupby('File')
valid_groups = {}
for name, g in groups:
    if len(g) >= TIME_STEP + ROLLOUT_STEPS:
        valid_groups[name] = g
    else:
        print(f'Skip group {name}, length={len(g)}')

print(f'Valid specimens: {len(valid_groups)}')

# =========================================================
# Build sliding-window rollout samples
# =========================================================
def create_samples_from_group(group_df, time_step, rollout_steps):
    group_df = group_df.sort_values('dt').reset_index(drop=True)
    dyn = group_df[dynamic_cols].values.astype(np.float32)
    sta = group_df[static_cols].iloc[0].values.astype(np.float32)
    tgt = group_df[target_col].values.astype(np.float32)
    dt_values = group_df['dt'].values.astype(np.float32)

    X_dyn, X_sta, Y, Future_dt = [], [], [], []
    for i in range(len(group_df) - time_step - rollout_steps + 1):
        x_dyn = dyn[i:i + time_step]
        future_slice = slice(i + time_step, i + time_step + rollout_steps)
        future_j = tgt[future_slice]
        prev_j = np.concatenate([[tgt[i + time_step - 1]], future_j[:-1]])
        y = future_j - prev_j
        future_dt = dt_values[future_slice]
        X_dyn.append(x_dyn)
        X_sta.append(sta)
        Y.append(y)
        Future_dt.append(future_dt)
    return np.array(X_dyn), np.array(X_sta), np.array(Y), np.array(Future_dt)


X_dyn_all, X_sta_all, Y_all, Future_dt_all = [], [], [], []
group_name_list = []
for name, group in valid_groups.items():
    xd, xs, y, fdt = create_samples_from_group(group, TIME_STEP, ROLLOUT_STEPS)
    if len(xd) == 0:
        continue
    X_dyn_all.append(xd)
    X_sta_all.append(xs)
    Y_all.append(y)
    Future_dt_all.append(fdt)
    group_name_list.extend([name] * len(xd))

X_dyn_all = np.concatenate(X_dyn_all, axis=0)
X_sta_all = np.concatenate(X_sta_all, axis=0)
Y_all = np.concatenate(Y_all, axis=0)
Future_dt_all = np.concatenate(Future_dt_all, axis=0)
group_name_all = np.array(group_name_list)

print(f'Total samples: {len(Y_all)}')
print(f'X_dyn shape: {X_dyn_all.shape}')
print(f'X_sta shape: {X_sta_all.shape}')
print(f'Y shape: {Y_all.shape}')
print(f'Future dt shape: {Future_dt_all.shape}')

# =========================================================
# Split specimens into train, validation, and test sets.
# Balance by final creep value so validation/test specimens are not drawn from a
# very different response range than the training specimens.
group_final_j = {
    name: group.sort_values('dt')[target_col].iloc[-1]
    for name, group in valid_groups.items()
}
unique_groups = sorted(valid_groups.keys(), key=lambda name: group_final_j[name])

n_total = len(unique_groups)
n_test = max(1, int(n_total * TEST_SPLIT))
n_val = max(1, int((n_total - n_test) * VAL_SPLIT))

train_groups, val_groups, test_groups = [], [], []
for rank, group_name in enumerate(unique_groups):
    bucket = rank % 5
    if bucket == 0 and len(test_groups) < n_test:
        test_groups.append(group_name)
    elif bucket == 1 and len(val_groups) < n_val:
        val_groups.append(group_name)
    else:
        train_groups.append(group_name)

for group_name in unique_groups:
    if len(test_groups) >= n_test and len(val_groups) >= n_val:
        break
    if group_name in train_groups:
        train_groups.remove(group_name)
        if len(test_groups) < n_test:
            test_groups.append(group_name)
        elif len(val_groups) < n_val:
            val_groups.append(group_name)

train_idx = np.where(np.isin(group_name_all, train_groups))[0]
val_idx = np.where(np.isin(group_name_all, val_groups))[0]
test_idx = np.where(np.isin(group_name_all, test_groups))[0]

print(f'Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}')
print(
    'Final Jcreep range | '
    f'Train: [{min(group_final_j[g] for g in train_groups):.3f}, {max(group_final_j[g] for g in train_groups):.3f}] | '
    f'Val: [{min(group_final_j[g] for g in val_groups):.3f}, {max(group_final_j[g] for g in val_groups):.3f}] | '
    f'Test: [{min(group_final_j[g] for g in test_groups):.3f}, {max(group_final_j[g] for g in test_groups):.3f}]'
)

# =========================================================
# Standardization statistics from training set only
train_dyn_flat = X_dyn_all[train_idx].reshape(-1, X_dyn_all.shape[2])
dyn_mean = train_dyn_flat.mean(axis=0)
dyn_std = train_dyn_flat.std(axis=0) + 1e-8

train_sta = X_sta_all[train_idx]
sta_mean = train_sta.mean(axis=0)
sta_std = train_sta.std(axis=0) + 1e-8

train_y = Y_all[train_idx]
y_mean = train_y.mean()
y_std = train_y.std() + 1e-8


def normalize_dyn(x):
    return (x - dyn_mean) / dyn_std


def normalize_sta(x):
    return (x - sta_mean) / sta_std


def normalize_y(x):
    return (x - y_mean) / y_std


def denormalize_y(x):
    return x * y_std + y_mean


X_dyn_norm = normalize_dyn(X_dyn_all)
X_sta_norm = normalize_sta(X_sta_all)
Y_norm = normalize_y(Y_all)
Future_dt_norm = (Future_dt_all - dyn_mean[0]) / dyn_std[0]


# =========================================================
# Dataset
# =========================================================
class CreepDataset(Dataset):
    def __init__(self, dyn, sta, y, future_dt):
        self.dyn = torch.tensor(dyn, dtype=torch.float32)
        self.sta = torch.tensor(sta, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.future_dt = torch.tensor(future_dt, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.dyn[idx], self.sta[idx], self.y[idx], self.future_dt[idx]


train_dataset = CreepDataset(X_dyn_norm[train_idx], X_sta_norm[train_idx], Y_norm[train_idx], Future_dt_norm[train_idx])
val_dataset = CreepDataset(X_dyn_norm[val_idx], X_sta_norm[val_idx], Y_norm[val_idx], Future_dt_norm[val_idx])
test_dataset = CreepDataset(X_dyn_norm[test_idx], X_sta_norm[test_idx], Y_norm[test_idx], Future_dt_norm[test_idx])
test_sample_groups = group_name_all[test_idx]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# =========================================================
# Model: SEGA-optimized TCN-LSTM-MultiHeadAttention + PINN constraints
# x_dyn: (batch, time_step, dyn_dim)
# x_sta: (batch, sta_dim)
# Model output: next-step delta-Jcreep, shape (batch,)
# =========================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + residual)


class TemporalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.2):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f'embed_dim={embed_dim} must be divisible by num_heads={num_heads}')

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        att_out, att_weight = self.attention(x, x, x, need_weights=True)
        att_out = self.norm(x + self.dropout(att_out))
        context = att_out[:, -1, :]
        return context, att_weight


class TCNAttLSTM(nn.Module):
    def __init__(
        self,
        dyn_dim,
        sta_dim,
        tcn_channels=128,
        lstm_hidden=128,
        lstm_layers=2,
        kernel_size=2,
        dropout=0.2,
        static_embed_dim=64,
        attention_heads=4
    ):
        super().__init__()
        self.static_embed_dim = static_embed_dim

        self.fc_static = nn.Sequential(
            nn.Linear(sta_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, static_embed_dim),
            nn.LayerNorm(static_embed_dim),
            nn.ReLU()
        )

        input_channels = dyn_dim
        self.tcn = nn.Sequential(
            TemporalBlock(input_channels, tcn_channels, kernel_size=kernel_size, dilation=1, dropout=dropout),
            TemporalBlock(tcn_channels, tcn_channels, kernel_size=kernel_size, dilation=2, dropout=dropout),
            TemporalBlock(tcn_channels, tcn_channels, kernel_size=kernel_size, dilation=4, dropout=dropout)
        )

        self.lstm = nn.LSTM(
            input_size=tcn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden)

        self.attention = TemporalMultiHeadAttention(
            embed_dim=lstm_hidden,
            num_heads=attention_heads,
            dropout=dropout
        )

        self.fc_out = nn.Sequential(
            nn.Linear(lstm_hidden + static_embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x_dyn, x_sta):
        static_embed = self.fc_static(x_sta)

        tcn_in = x_dyn.transpose(1, 2)
        tcn_out = self.tcn(tcn_in)
        seq_out = tcn_out.transpose(1, 2)

        lstm_out, _ = self.lstm(seq_out)
        lstm_out = self.lstm_norm(lstm_out)
        att_context, _ = self.attention(lstm_out)

        fusion = torch.cat([att_context, static_embed], dim=1)
        out = self.fc_out(fusion)
        return out.squeeze(-1)

# =========================================================
# Physics-informed loss functions
# =========================================================
def monotonicity_loss(pred, target):
    """Penalize negative predicted creep increments."""
    if pred.ndim == 2 and pred.size(1) > 1:
        zero_delta_norm = (0.0 - y_mean) / y_std
        return torch.relu(zero_delta_norm - pred).mean()
    diff = pred - target
    penalty = torch.relu(-diff)
    return penalty.mean()


def smoothness_loss(pred, target):
    """Penalize strong oscillation in recursive prediction."""
    if pred.ndim == 2 and pred.size(1) > 2:
        curvature = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        return (curvature ** 2).mean()
    error = pred - target
    return (error ** 2).mean()


def rollout_predict(model, batch_dyn, batch_sta, future_dt_norm):
    current_seq = batch_dyn.clone()
    preds = []
    idx_dt = 0
    idx_log_dt = 1
    idx_j = 2
    idx_dj_dt = 3

    for step in range(future_dt_norm.size(1)):
        pred_norm = model(current_seq, batch_sta)
        preds.append(pred_norm)

        pred_delta_raw = torch.relu(pred_norm * y_std + y_mean)
        last_step = current_seq[:, -1, :]
        last_dt_raw = last_step[:, idx_dt] * dyn_std[idx_dt] + dyn_mean[idx_dt]
        last_j_raw = last_step[:, idx_j] * dyn_std[idx_j] + dyn_mean[idx_j]
        new_dt_raw = future_dt_norm[:, step] * dyn_std[idx_dt] + dyn_mean[idx_dt]
        delta_dt_raw = torch.clamp(new_dt_raw - last_dt_raw, min=1e-6)
        new_j_raw = last_j_raw + pred_delta_raw
        new_log_dt_raw = torch.log1p(torch.clamp(new_dt_raw, min=0))
        new_dj_dt_raw = pred_delta_raw / delta_dt_raw

        new_step = current_seq[:, -1:, :].clone()
        new_step[:, 0, idx_dt] = future_dt_norm[:, step]
        new_step[:, 0, idx_log_dt] = (new_log_dt_raw - dyn_mean[idx_log_dt]) / dyn_std[idx_log_dt]
        new_step[:, 0, idx_j] = (new_j_raw - dyn_mean[idx_j]) / dyn_std[idx_j]
        new_step[:, 0, idx_dj_dt] = (new_dj_dt_raw - dyn_mean[idx_dj_dt]) / dyn_std[idx_dj_dt]
        current_seq = torch.cat([current_seq[:, 1:, :], new_step], dim=1)

    return torch.stack(preds, dim=1)


def rollout_pinn_loss(output, target, criterion):
    data_loss = criterion(output, target)
    mono_loss = monotonicity_loss(output, target)
    smooth_loss = smoothness_loss(output, target)
    step_weights = torch.linspace(1.0, 0.7, output.size(1), device=output.device).view(1, -1)
    growth_weights = 1.0 + torch.clamp(torch.mean(torch.abs(target), dim=1, keepdim=True), 0.0, 2.0)
    weighted_data_loss = (torch.abs(output - target) * step_weights * growth_weights).mean()
    return (data_loss
            + 0.5 * weighted_data_loss
            + LAMBDA_PHYS_MONO * mono_loss
            + LAMBDA_PHYS_SMOOTH * smooth_loss)


# =========================================================
# Model input dimensions
dyn_dim = len(dynamic_cols)
sta_dim = len(static_cols)


def sega_build_model(params):
    return TCNAttLSTM(
        dyn_dim=dyn_dim,
        sta_dim=sta_dim,
        tcn_channels=int(params['hidden_size']),
        lstm_hidden=int(params['hidden_size']),
        lstm_layers=int(params['num_layers']),
        dropout=float(params['dropout'])
    ).to(device)


def sega_pinn_loss(output, target, criterion):
    data_loss = criterion(output, target)
    mono_loss = monotonicity_loss(output, target)
    smooth_loss = smoothness_loss(output, target)
    return (data_loss
            + LAMBDA_PHYS_MONO * mono_loss
            + LAMBDA_PHYS_SMOOTH * smooth_loss)


def sega_fitness(params):
    candidate_model = sega_build_model(params)
    candidate_criterion = nn.SmoothL1Loss()
    candidate_optimizer = optim.AdamW(
        candidate_model.parameters(),
        lr=float(params['lr']),
        weight_decay=1e-4
    )

    for _ in range(SEGA_FITNESS_EPOCHS):
        candidate_model.train()
        for batch_dyn, batch_sta, batch_y, batch_future_dt in train_loader:
            batch_dyn = batch_dyn.to(device)
            batch_sta = batch_sta.to(device)
            batch_y = batch_y.to(device)
            batch_future_dt = batch_future_dt.to(device)

            candidate_optimizer.zero_grad()
            output = rollout_predict(candidate_model, batch_dyn, batch_sta, batch_future_dt)
            loss = rollout_pinn_loss(output, batch_y, candidate_criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate_model.parameters(), 1.0)
            candidate_optimizer.step()

    candidate_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_dyn, batch_sta, batch_y, batch_future_dt in val_loader:
            batch_dyn = batch_dyn.to(device)
            batch_sta = batch_sta.to(device)
            batch_y = batch_y.to(device)
            batch_future_dt = batch_future_dt.to(device)
            output = rollout_predict(candidate_model, batch_dyn, batch_sta, batch_future_dt)
            val_loss += rollout_pinn_loss(output, batch_y, candidate_criterion).item()

    return val_loss / max(1, len(val_loader))


class SelfAdaptiveEvolutionaryGeneticAlgorithm:
    def __init__(self, pop_size, generations, elite_size):
        self.pop_size = pop_size
        self.generations = generations
        self.elite_size = elite_size
        self.hidden_choices = [64, 96, 128]
        self.layer_choices = [1, 2]
        self.dropout_bounds = (0.20, 0.40)
        self.lr_bounds = (2e-4, 1e-3)

    def random_individual(self):
        return {
            'hidden_size': random.choice(self.hidden_choices),
            'num_layers': random.choice(self.layer_choices),
            'dropout': random.uniform(*self.dropout_bounds),
            'lr': 10 ** random.uniform(np.log10(self.lr_bounds[0]), np.log10(self.lr_bounds[1])),
            'mutation_rate': random.uniform(0.10, 0.35),
            'crossover_rate': random.uniform(0.55, 0.90),
            'fitness': None
        }

    def params(self, individual):
        return {
            'hidden_size': int(individual['hidden_size']),
            'num_layers': int(individual['num_layers']),
            'dropout': float(individual['dropout']),
            'lr': float(individual['lr'])
        }

    def evaluate(self, individual):
        individual['fitness'] = sega_fitness(self.params(individual))
        return individual['fitness']

    def select(self, population, k=3):
        candidates = random.sample(population, min(k, len(population)))
        return min(candidates, key=lambda item: item['fitness'])

    def crossover(self, parent_a, parent_b):
        if random.random() > max(parent_a['crossover_rate'], parent_b['crossover_rate']):
            return parent_a.copy()

        child = {}
        for key in ['hidden_size', 'num_layers', 'dropout', 'lr']:
            child[key] = parent_a[key] if random.random() < 0.5 else parent_b[key]
        child['mutation_rate'] = (parent_a['mutation_rate'] + parent_b['mutation_rate']) / 2
        child['crossover_rate'] = (parent_a['crossover_rate'] + parent_b['crossover_rate']) / 2
        child['fitness'] = None
        return child

    def mutate(self, individual):
        child = individual.copy()

        # Self-adaptive rates: mutation and crossover probabilities evolve too.
        child['mutation_rate'] = float(np.clip(
            child['mutation_rate'] * np.exp(np.random.normal(0, 0.15)), 0.05, 0.60
        ))
        child['crossover_rate'] = float(np.clip(
            child['crossover_rate'] * np.exp(np.random.normal(0, 0.10)), 0.30, 0.95
        ))

        if random.random() < child['mutation_rate']:
            child['hidden_size'] = random.choice(self.hidden_choices)
        if random.random() < child['mutation_rate']:
            child['num_layers'] = random.choice(self.layer_choices)
        if random.random() < child['mutation_rate']:
            child['dropout'] = float(np.clip(
                child['dropout'] + np.random.normal(0, 0.05),
                self.dropout_bounds[0],
                self.dropout_bounds[1]
            ))
        if random.random() < child['mutation_rate']:
            child['lr'] = float(np.clip(
                child['lr'] * np.exp(np.random.normal(0, 0.45)),
                self.lr_bounds[0],
                self.lr_bounds[1]
            ))

        child['fitness'] = None
        return child

    def run(self):
        population = [self.random_individual() for _ in range(self.pop_size)]
        best = None

        for gen in range(self.generations):
            print(f'\nSEGA generation {gen + 1}/{self.generations}')
            for idx, individual in enumerate(population):
                if individual['fitness'] is None:
                    fitness = self.evaluate(individual)
                    print(f'  individual {idx + 1:02d} | val_loss={fitness:.6f} | {self.params(individual)}')

            population = sorted(population, key=lambda item: item['fitness'])
            best = population[0].copy()
            print(f'  best fitness={best["fitness"]:.6f} | params={self.params(best)}')

            next_population = [item.copy() for item in population[:self.elite_size]]
            while len(next_population) < self.pop_size:
                parent_a = self.select(population)
                parent_b = self.select(population)
                child = self.crossover(parent_a, parent_b)
                next_population.append(self.mutate(child))
            population = next_population

        return self.params(best), best['fitness']


if SKIP_TRAINING:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f'SKIP_TRAINING=True, but model file was not found: {MODEL_PATH}. '
            'Train once first or set SKIP_TRAINING=False.'
        )
    if not os.path.exists(BEST_PARAMS_PATH):
        raise FileNotFoundError(
            f'SKIP_TRAINING=True, but params file was not found: {BEST_PARAMS_PATH}. '
            'Train once first with SKIP_TRAINING=False so the script can save matched hyperparameters.'
        )
    with open(BEST_PARAMS_PATH, 'r', encoding='utf-8-sig') as f:
        best_sega_params = json.load(f)
    print(f'\nSkip training. Loaded SEGA params from {BEST_PARAMS_PATH}: {best_sega_params}')
else:
    print('\nSKIP_TRAINING=False: run SEGA search and train a new model.')
    sega = SelfAdaptiveEvolutionaryGeneticAlgorithm(
        pop_size=SEGA_POP_SIZE,
        generations=SEGA_GENERATIONS,
        elite_size=SEGA_ELITE_SIZE
    )
    best_sega_params, best_sega_fitness = sega.run()
    with open(BEST_PARAMS_PATH, 'w', encoding='utf-8') as f:
        json.dump(best_sega_params, f, indent=2)
    print(f'\nSEGA best params: {best_sega_params}')
    print(f'SEGA best validation loss: {best_sega_fitness:.6f}')
    print(f'Saved SEGA params to: {BEST_PARAMS_PATH}')

HIDDEN_SIZE = best_sega_params['hidden_size']
NUM_LAYERS = best_sega_params['num_layers']
DROPOUT = best_sega_params['dropout']
LR = best_sega_params['lr']

model = TCNAttLSTM(
    dyn_dim=dyn_dim,
    sta_dim=sta_dim,
    tcn_channels=HIDDEN_SIZE,
    lstm_hidden=HIDDEN_SIZE,
    lstm_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

print(f'\nModel parameters: {sum(p.numel() for p in model.parameters()):,}')

# =========================================================
# Training loop
# =========================================================
best_val_loss = float('inf')
patience_counter = 0
train_losses, val_losses = [], []

if SKIP_TRAINING:
    print('Skip training loop. Go directly to evaluation and prediction.')

for epoch in range(0 if SKIP_TRAINING else EPOCHS):
    # ===== Train =====
    model.train()
    train_loss = 0
    for batch_dyn, batch_sta, batch_y, batch_future_dt in train_loader:
        batch_dyn = batch_dyn.to(device)
        batch_sta = batch_sta.to(device)
        batch_y = batch_y.to(device)
        batch_future_dt = batch_future_dt.to(device)

        optimizer.zero_grad()
        output = rollout_predict(model, batch_dyn, batch_sta, batch_future_dt)

        loss = rollout_pinn_loss(output, batch_y, criterion)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # ===== Validation =====
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_dyn, batch_sta, batch_y, batch_future_dt in val_loader:
            batch_dyn = batch_dyn.to(device)
            batch_sta = batch_sta.to(device)
            batch_y = batch_y.to(device)
            batch_future_dt = batch_future_dt.to(device)

            output = rollout_predict(model, batch_dyn, batch_sta, batch_future_dt)
            loss = rollout_pinn_loss(output, batch_y, criterion)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    # ===== Early Stopping =====
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}')

if not SKIP_TRAINING:
    print('Training finished.')
    print(f'Saved best model to: {MODEL_PATH}')

# =========================================================
# Load best model and evaluate on test set
# =========================================================
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_preds, all_targets = [], []
with torch.no_grad():
    for batch_dyn, batch_sta, batch_y, batch_future_dt in test_loader:
        batch_dyn = batch_dyn.to(device)
        batch_sta = batch_sta.to(device)
        batch_future_dt = batch_future_dt.to(device)
        output = rollout_predict(model, batch_dyn, batch_sta, batch_future_dt)
        all_preds.append(output.cpu().numpy())
        all_targets.append(batch_y.numpy())

pred_deltas = denormalize_y(np.concatenate(all_preds))
target_deltas = denormalize_y(np.concatenate(all_targets))
last_j_test = X_dyn_all[test_idx, -1, dynamic_cols.index('Jcreep')]
preds = last_j_test[:, None] + np.cumsum(pred_deltas, axis=1)
targets = last_j_test[:, None] + np.cumsum(target_deltas, axis=1)

if shift > 0:
    preds -= shift
    targets -= shift

# =========================================================
# Evaluation metrics
# =========================================================
preds_eval = preds.reshape(-1)
targets_eval = targets.reshape(-1)
r2 = r2_score(targets_eval, preds_eval)
rmse = np.sqrt(np.mean((preds_eval - targets_eval) ** 2))
mape = mean_absolute_percentage_error(targets_eval, preds_eval) * 100

print(f'\n{"="*40}')
print(f'Test R2   : {r2:.4f}')
print(f'Test RMSE : {rmse:.4f}')
print(f'Test MAPE : {mape:.2f}%')
print(f'{"="*40}')

# =========================================================
# Test result visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.ravel()

# Scatter plot
ax = axes[0]
ax.scatter(targets_eval, preds_eval, alpha=0.4, s=10)
min_v = min(targets_eval.min(), preds_eval.min())
max_v = max(targets_eval.max(), preds_eval.max())
ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=1.5)
ax.set_xlabel('True Jcreep')
ax.set_ylabel('Predicted Jcreep')
ax.set_title(f'Scatter Plot (R{r2:.4f})')
ax.grid(True, alpha=0.3)

# Residual distribution
ax = axes[1]
residuals = preds - targets
residuals = residuals.reshape(-1)
ax.hist(residuals, bins=50, edgecolor='k', alpha=0.7)
ax.axvline(0, color='r', linestyle='--')
ax.set_xlabel('Residual (Pred - True)')
ax.set_ylabel('Frequency')
ax.set_title('Residual Distribution')
ax.grid(True, alpha=0.3)

# Loss curves
ax = axes[2]
ax.plot(train_losses, label='Train Loss', alpha=0.8)
ax.plot(val_losses, label='Val Loss', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Curves')
ax.legend()
ax.grid(True, alpha=0.3)

# True vs predicted sequence plot
ax = axes[3]
sample_idx = np.arange(len(targets_eval))
ax.plot(sample_idx, targets_eval, label='True', linewidth=1.5)
ax.plot(sample_idx, preds_eval, label='Pred', linewidth=1.5)
ax.set_xlabel('Sample')
ax.set_ylabel('Jcreep')
ax.set_title(f'Recursive Rollout Prediction ({ROLLOUT_STEPS} steps)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SEGA_TCN_ATT_LSTM_PINN_results.png', dpi=150)
plt.show()


# =========================================================
# Recursive multi-step mode
# Feed predicted values back into the input window for future steps.
# This block does not run one-step diagnostic prediction.
# =========================================================
print('\\nThis script runs recursive multi-step prediction only.')


# =========================================================
# Recursive multi-step forecast function
# =========================================================
def recursive_multistep_predict(model, initial_dyn, static_feat, future_steps,
                                dyn_mean, dyn_std, y_mean, y_std, shift=0,
                                future_dt_values=None):
    """
    Recursive multi-step prediction.
    1. Predict the next delta-Jcreep from the current input window.
    2. Add the predicted increment to the previous cumulative Jcreep.
    3. Update dt, log_dt, Jcreep, and dJ_dt in the input window.
    4. Slide the window and repeat.

    Args:
        model: Trained TCNAttLSTM model.
        initial_dyn: Standardized input window, shape (1, TIME_STEP, n_dynamic_features).
        static_feat: Standardized static features, shape (1, n_static_features).
        future_steps: Number of future points to forecast.
        dyn_mean, dyn_std: Dynamic-feature normalization statistics.
        y_mean, y_std: Delta-target normalization statistics.
        shift: Offset used when original Jcreep has negative values.
        future_dt_values: Real future dt values. If None, dt increases by 1 each step.

    Returns:
        Cumulative Jcreep predictions after inverse scaling, shape (future_steps,).
    """
    model.eval()
    current_seq = initial_dyn.clone()
    predictions = []

    with torch.no_grad():
        if future_dt_values is not None:
            future_dt_values = np.asarray(future_dt_values, dtype=np.float32)
            future_steps = len(future_dt_values)

        for step in range(future_steps):
            pred_norm = model(current_seq, static_feat)
            pred_delta_raw = max(pred_norm.item() * y_std + y_mean, 0.0)

            last_step = current_seq[:, -1, :].clone()
            idx_dt = 0
            idx_log_dt = 1
            idx_j = 2
            idx_dj_dt = 3

            last_dt_raw = last_step[0, idx_dt].item() * dyn_std[idx_dt] + dyn_mean[idx_dt]
            last_j_raw = last_step[0, idx_j].item() * dyn_std[idx_j] + dyn_mean[idx_j]

            if future_dt_values is None:
                new_dt_raw = last_dt_raw + 1
            else:
                new_dt_raw = float(future_dt_values[step])

            delta_dt_raw = max(new_dt_raw - last_dt_raw, 1e-6)
            new_j_raw = last_j_raw + pred_delta_raw
            new_log_dt_raw = np.log1p(max(new_dt_raw, 0))
            new_dj_dt_raw = pred_delta_raw / delta_dt_raw
            predictions.append(new_j_raw)

            new_step_raw = np.array([
                new_dt_raw, new_log_dt_raw, new_j_raw, new_dj_dt_raw
            ], dtype=np.float32)

            new_step_norm = (new_step_raw - dyn_mean) / dyn_std
            new_step_tensor = torch.tensor(new_step_norm, dtype=torch.float32).view(1, 1, -1).to(current_seq.device)

            current_seq = torch.cat([current_seq[:, 1:, :], new_step_tensor], dim=1)

    predictions_raw = np.array(predictions)
    if shift > 0:
        predictions_raw -= shift
    return predictions_raw


# =========================================================
# Test recursive multi-step prediction
# =========================================================
FUTURE_STEPS = 15

# Select one test sample and find its specimen group.
sample_idx = 0
sample_group_name = test_sample_groups[sample_idx]
sample_dyn, sample_sta, sample_y, sample_future_dt = test_dataset[sample_idx]
sample_dyn = sample_dyn.unsqueeze(0).to(device)  # (1, TIME_STEP, n_dyn)
sample_sta = sample_sta.unsqueeze(0).to(device)   # (1, n_sta)

# Load the full specimen curve for comparison with recursive forecasts.
if sample_group_name not in valid_groups:
    raise KeyError(f'Sample group {sample_group_name} was not found in valid_groups.')
sample_group = valid_groups[sample_group_name].sort_values('dt').reset_index(drop=True)
sample_t = sample_group['dt'].values
sample_j_true = sample_group[target_col].values
if shift > 0:
    sample_j_true = sample_j_true - shift

# Use the actual last dt of the selected input window.
init_t = float(sample_dyn[0, -1, 0].item() * dyn_std[0] + dyn_mean[0])
future_t = sample_t[sample_t > init_t][:FUTURE_STEPS]
if len(future_t) == 0:
    raise ValueError('Selected sample has no future dt values for recursive prediction.')

print(f'\nRecursive forecasting {len(future_t)} real future dt points...')
future_preds = recursive_multistep_predict(
    model, sample_dyn, sample_sta, len(future_t),
    dyn_mean, dyn_std, y_mean, y_std, shift,
    future_dt_values=future_t
)

print(f'Prediction range: [{future_preds.min():.2f}, {future_preds.max():.2f}]')

# Plot recursive multi-step forecasts with one specimen per subplot.
test_group_names = []
test_dataset_indices = []
for ds_idx, group_name in enumerate(test_sample_groups):
    if group_name not in test_group_names:
        test_group_names.append(group_name)
        test_dataset_indices.append(ds_idx)
    if len(test_group_names) >= 5:
        break

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()
fig.suptitle('Recursive Multi-step Prediction Check', fontsize=14)

for ax_idx, ds_idx in enumerate(test_dataset_indices):
    ax = axes[ax_idx]
    sd, ss, _, _ = test_dataset[ds_idx]
    sd = sd.unsqueeze(0).to(device)
    ss = ss.unsqueeze(0).to(device)
    sample_group_name = test_group_names[ax_idx]
    group_df = valid_groups[sample_group_name].sort_values('dt')
    group_dt = group_df['dt'].values
    group_true = group_df[target_col].values
    if shift > 0:
        group_true = group_true - shift
    init_dt = float(sd[0, -1, 0].item() * dyn_std[0] + dyn_mean[0])
    future_mask = group_dt > init_dt
    group_future_t = group_dt[future_mask]
    if len(group_future_t) == 0:
        ax.axis('off')
        continue
    fp = recursive_multistep_predict(
        model, sd, ss, len(group_future_t), dyn_mean, dyn_std, y_mean, y_std, shift,
        future_dt_values=group_future_t
    )

    ax.plot(group_dt, group_true, 'o-', color='tab:green', markersize=3,
            linewidth=1.5, label='True')
    ax.plot(group_future_t, fp, 'o--', color='red', markersize=3,
            linewidth=1.5, alpha=0.85, label='Recursive Pred')
    ax.axvline(init_dt, color='gray', linestyle=':', alpha=0.45)
    ax.set_title(sample_group_name)
    ax.set_xlabel('dt (days)')
    ax.set_ylabel('Jcreep')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

for ax in axes[len(test_dataset_indices):]:
    ax.axis('off')

plt.tight_layout()
plt.savefig('SEGA_TCN_ATT_LSTM_PINN_recursive.png', dpi=300)
plt.show()

print('\nDone. Models and figures saved.')

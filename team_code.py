#!/usr/bin/env python

# team_code.py for PhysioNet Challenge 2025 - Chagas Disease Detection
# Author: Saber Jelodari, Adam Bokun
# Date 4th July 2025

import os
import pywt
import random
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.signal as sgn
from tqdm import tqdm
from skimage.transform import resize

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

from helper_code import (
    find_records,
    load_signals,
    load_label,
    load_header,
    get_age,
    get_sex,
)



# Hyperparameters & Config

HP = {
    # CNN parameters
    "cnn_channels": [12, 32, 64, 128],
    "dropout_rate": 0.3,

    # LSTM parameters
    "rnn_hidden_size": 64,
    "rnn_num_layers": 2,

    # Fully-connected parameters
    "fc_size": 128,

    # Data parameters
    "desired_len": 4000,
    "random_crop": True,

    # Training parameters
    "learning_rate": 0.001,
    "batch_size": 16,
    "epochs": 20,
    "weight_decay": 1e-5,  # L2 regularization

    # Class weighting (balance)
    "pos_weight": 2.0,

    # Data augmentation probabilities
    "augment_time_shift": 0.5,
    "max_time_shift": 200,
    "augment_amp_scaling": 0.5,
    "amp_scale_range": [0.9, 1.1],
    "augment_lead_dropout": 0.1,

    # Early stopping
    "early_stopping_patience": 5,

    # Decision threshold
    "decision_threshold": 0.5,
}

# Will store the best threshold found by the training loop for final inference
BEST_THRESHOLD = 0.5


# Device selection

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Model Definition

class WaveletECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

def generate_scalogram(ecg_signal, scales=np.arange(1, 128), waveletname='morl', target_size=(128, 128)):
    coef, _ = pywt.cwt(ecg_signal, scales, waveletname)
    scalogram = np.abs(coef)
    range_val = scalogram.max() - scalogram.min()
    if range_val > 1e-6:
        scalogram = (scalogram - scalogram.min()) / range_val
    else:
        scalogram = np.zeros_like(scalogram)
    scalogram_resized = resize(scalogram, target_size, mode='reflect', anti_aliasing=True)
    return scalogram_resized


def ecg_to_scalogram_tensor(signals, fs=400, target_size=(128, 128)):
    signals = resample_to_400Hz(signals, fs)
    signals = signals[:, :12]
    scalograms = [generate_scalogram(signals[:, i], target_size=target_size) for i in range(12)]
    stacked = np.stack(scalograms, axis=0)
    return torch.tensor(stacked, dtype=torch.float32)


def process_ecg_wavelet(signals, fields):
    fs = fields.get('fs', 400)
    return ecg_to_scalogram_tensor(signals, fs)

class DynamicCostSensitiveLoss(nn.Module):
    def __init__(self, total_class_counts, device):
        super().__init__()
        self.device = device
        self.total_counts = total_class_counts
        self.eps = 1e-6

    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        # Counts per batch
        y_true = labels.detach()
        y_pred = preds.detach()

        tp = ((y_true == 1) & (y_pred == 1)).sum().float()
        tn = ((y_true == 0) & (y_pred == 0)).sum().float()
        fp = ((y_true == 0) & (y_pred == 1)).sum().float()
        fn = ((y_true == 1) & (y_pred == 0)).sum().float()

        # FPR and FNR
        fpr = fp / (fp + tn + self.eps)
        fnr = fn / (fn + tp + self.eps)
        harmonic_fpr_fnr = 2 * (fpr * fnr) / (fpr + fnr + self.eps)

        # Class weights
        batch_pos = (labels == 1).sum().item()
        batch_neg = (labels == 0).sum().item()
        batch_total = batch_pos + batch_neg + self.eps

        batch_pos_weight = batch_neg / batch_total
        batch_neg_weight = batch_pos / batch_total

        total_pos = self.total_counts[1]
        total_neg = self.total_counts[0]
        total_total = total_pos + total_neg + self.eps

        total_pos_weight = total_neg / total_total
        total_neg_weight = total_pos / total_total

        # Final class weights
        pos_weight = ((batch_pos_weight ** 2 + total_pos_weight ** 2) / 2) ** 0.5
        neg_weight = ((batch_neg_weight ** 2 + total_neg_weight ** 2) / 2) ** 0.5

        # Final misclassification cost
        misclass_cost = (pos_weight + harmonic_fpr_fnr)

        # Compute weighted loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        weights = torch.where(labels == 1, pos_weight, neg_weight)
        weighted_loss = misclass_cost * weights.to(self.device) * bce_loss

        return weighted_loss.mean()

def resample_to_400Hz(signals, original_fs):
    """Resample signals to 400 Hz if needed."""
    if original_fs == 0:
        original_fs = 400  # fallback to 400 if unknown
    if original_fs == 400:
        return signals
    # Use sci
    new_len = int(signals.shape[0] * 400 / original_fs)
    signals = sgn.resample(signals, new_len, axis=0)
    return signals

def random_time_shift(ecg):

    max_shift = HP["max_time_shift"]
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return ecg
    # shift > 0 => shift to the right
    # shift < 0 => shift to the left
    ecg_shifted = torch.zeros_like(ecg)
    if shift > 0:
        ecg_shifted[:, shift:] = ecg[:, :-shift]
    else:
        ecg_shifted[:, :shift] = ecg[:, -shift:]
    return ecg_shifted

def random_amplitude_scaling(ecg):

    scale_min, scale_max = HP["amp_scale_range"]
    scale = random.uniform(scale_min, scale_max)
    return ecg * scale

def random_lead_dropout(ecg):

    lead_to_drop = random.randint(0, ecg.shape[0] - 1)
    ecg[lead_to_drop, :] = 0.0
    return ecg

def process_ecg(signals, fields, augment=False):

    fs = fields.get('fs', 400)
    signals = resample_to_400Hz(signals, fs)

    num_leads = signals.shape[1]
    # If fewer than 12 leads, pad; if more, slice
    if num_leads < 12:
        tmp = np.zeros((signals.shape[0], 12), dtype=signals.dtype)
        tmp[:, :num_leads] = signals
        signals = tmp
    elif num_leads > 12:
        signals = signals[:, :12]

    ecg = torch.tensor(signals.T, dtype=torch.float32)  # shape: (12, time)

    desired_len = HP["desired_len"]
    cur_len = ecg.shape[1]

    # If we want to random-crop signals that are too long:
    if cur_len > desired_len:
        if HP["random_crop"] and augment:
            # random start
            max_start = cur_len - desired_len
            start = random.randint(0, max_start)
        else:
            # center crop
            start = (cur_len - desired_len) // 2
        ecg = ecg[:, start:start + desired_len]
    elif cur_len < desired_len:
        # zero-pad at the end
        pad = torch.zeros((12, desired_len))
        pad[:, :cur_len] = ecg
        ecg = pad

    # Data augmentation
    if augment:
        # random time shift
        if random.random() < HP["augment_time_shift"]:
            ecg = random_time_shift(ecg)
        # random amplitude scaling
        if random.random() < HP["augment_amp_scaling"]:
            ecg = random_amplitude_scaling(ecg)
        # random lead dropout
        if random.random() < HP["augment_lead_dropout"]:
            ecg = random_lead_dropout(ecg)

    # Normalize
    mean = ecg.mean(dim=1, keepdim=True)
    std = ecg.std(dim=1, keepdim=True) + 1e-6
    ecg = (ecg - mean) / std

    return ecg

def calculate_metrics(y_true, y_pred, y_prob):
    """Return a dict of metrics."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['sensitivity'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except:
        metrics['auc'] = 0.5
    return metrics


# Required Challenge Functions

def train_model(data_folder, model_folder, verbose):
    global BEST_THRESHOLD
    device = get_device()
    if verbose:
        print(f"[train_model] Using device: {device}")

    # 1) Find records
    records = find_records(data_folder)
    if verbose:
        print(f"[train_model] Found {len(records)} records in {data_folder}.")

    # 2) Collect valid records
    valid_records = []
    for record in records:
        record_path = os.path.join(data_folder, record)
        header_file = record_path + '.hea'
        if not os.path.isfile(header_file):
            if verbose:
                print(f"Warning: Missing header {header_file}")
            continue
        try:
            signals, fields = load_signals(record_path)
            label = load_label(record_path)
            valid_records.append((record_path, label))
        except Exception as e:
            if verbose:
                print(f"Skipping {record}: {str(e)}")

    if len(valid_records) == 0:
        raise ValueError("No valid records found. Check data folder structure.")

    positives = sum(1 for _, lbl in valid_records if lbl == 1)
    negatives = len(valid_records) - positives
    if verbose:
        print(f"Valid records: {len(valid_records)}; Positives: {positives}, Negatives: {negatives}")

    # 3) Simple balancing strategy
    pos_records = [(p, l) for (p, l) in valid_records if l == 1]
    neg_records = [(p, l) for (p, l) in valid_records if l == 0]

    # Example approach: keep up to 2× the smaller class
    if len(pos_records) < len(neg_records):
        neg_records = random.sample(neg_records, min(len(pos_records) * 2, len(neg_records)))
    else:
        pos_records = random.sample(pos_records, min(len(neg_records) * 2, len(pos_records)))

    balanced_records = pos_records + neg_records
    random.shuffle(balanced_records)

    # 4) Train/validation split
    split_idx = int(0.8 * len(balanced_records))
    train_records = balanced_records[:split_idx]
    val_records = balanced_records[split_idx:]
    if verbose:
        print(f"Training set: {len(train_records)}, Validation set: {len(val_records)}")

    # For saving confusion matrix each epoch in CSV
    csv_cm_path = os.path.join(model_folder, "confusion_matrix.csv")
    # Write header (overwrite if it already exists)
    with open(csv_cm_path, 'w') as f:
        f.write("epoch,tn,fp,fn,tp\n")

    # 5) Create model, optimizer, and BCE with logit + pos_weight
    model = WaveletECGModel().to(device)
    total_class_counts = {
        0: negatives,
        1: positives
    }
    criterion = DynamicCostSensitiveLoss(total_class_counts, device)
    optimizer = optim.Adam(model.parameters(), lr=HP["learning_rate"], weight_decay=HP["weight_decay"])

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    batch_size = HP["batch_size"]

    for epoch in range(1, HP["epochs"] + 1):
        # ---------------- TRAINING ----------------
        model.train()
        random.shuffle(train_records)
        running_loss = 0.0

        for i in range(0, len(train_records), batch_size):
            batch = train_records[i:i + batch_size]
            batch_ecgs = []
            batch_labels = []

            for rec_path, lbl in batch:
                try:
                    signals, fields = load_signals(rec_path)
                    ecg = process_ecg_wavelet(signals, fields)  # apply augmentation
                    batch_ecgs.append(ecg)
                    batch_labels.append(lbl)
                except:
                    continue

            if len(batch_ecgs) == 0:
                continue

            ecgs_t = torch.stack(batch_ecgs).to(device)
            labels_t = torch.tensor(batch_labels, dtype=torch.float32).view(-1, 1).to(device)

            optimizer.zero_grad()
            logits = model(ecgs_t)
            loss = criterion(logits, labels_t)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_ecgs)

        avg_train_loss = running_loss / len(train_records)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        y_true, y_scores = [], []

        with torch.no_grad():
            for i in range(0, len(val_records), batch_size):
                batch = val_records[i:i + batch_size]
                batch_ecgs = []
                batch_labels = []

                for rec_path, lbl in batch:
                    try:
                        signals, fields = load_signals(rec_path)
                        ecg = process_ecg_wavelet(signals, fields)
                        batch_ecgs.append(ecg)
                        batch_labels.append(lbl)
                    except:
                        continue

                if not batch_ecgs:
                    continue

                ecgs_t = torch.stack(batch_ecgs).to(device)
                labels_t = torch.tensor(batch_labels, dtype=torch.float32).view(-1, 1).to(device)

                logits = model(ecgs_t)
                batch_loss = criterion(logits, labels_t)
                val_loss += batch_loss.item() * len(batch_ecgs)

                prob = torch.sigmoid(logits).cpu().numpy().flatten()
                y_true.extend(batch_labels)
                y_scores.extend(prob)

        avg_val_loss = val_loss / len(val_records) if val_records else 0

        # Search best threshold on validation set to maximize F1
        best_f1_for_epoch = -1
        best_thresh_for_epoch = HP["decision_threshold"]
        if y_true:
            thresholds_to_try = np.linspace(0.0, 1.0, 51)
            for t in thresholds_to_try:
                preds_t = [1 if s >= t else 0 for s in y_scores]
                f1_t = f1_score(y_true, preds_t, zero_division=0)
                if f1_t > best_f1_for_epoch:
                    best_f1_for_epoch = f1_t
                    best_thresh_for_epoch = t

        # Compute final preds for logging
        preds_val = [1 if s >= best_thresh_for_epoch else 0 for s in y_scores]
        metrics_val = calculate_metrics(y_true, preds_val, y_scores)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, preds_val, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()

        # Save confusion matrix row in CSV
        with open(csv_cm_path, 'a') as f:
            f.write(f"{epoch},{tn},{fp},{fn},{tp}\n")

        if verbose:
            print(f"Epoch {epoch}/{HP['epochs']} | "
                  f"TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | "
                  f"F1={metrics_val['f1']:.4f} | AUC={metrics_val['auc']:.4f} | "
                  f"BestThresh={best_thresh_for_epoch:.3f}")
            print(f"Confusion Matrix => TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            BEST_THRESHOLD = best_thresh_for_epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= HP["early_stopping_patience"]:
                if verbose:
                    print("Early stopping triggered.")
                break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 6) Save model
    os.makedirs(model_folder, exist_ok=True)
    torch.save({'model_state': model.state_dict()}, os.path.join(model_folder, 'model.pth'))

    # Also save with joblib (optional)
    model_dict = {'model': model}
    joblib.dump(model_dict, os.path.join(model_folder, 'model.sav'))

    with open(os.path.join(model_folder, 'best_threshold.txt'), 'w') as f:
        f.write(str(BEST_THRESHOLD))

    if verbose:
        print(f"Model saved to {model_folder}, best val threshold = {BEST_THRESHOLD:.3f}")
        print(f"Confusion matrix CSV saved to: {csv_cm_path}")

def load_model(model_folder, verbose):
    global BEST_THRESHOLD
    device = get_device()
    model_path = os.path.join(model_folder, 'model.pth')
    joblib_path = os.path.join(model_folder, 'model.sav')
    threshold_path = os.path.join(model_folder, 'best_threshold.txt')

    # Attempt to load threshold
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            val = f.read().strip()
            try:
                BEST_THRESHOLD = float(val)
            except:
                BEST_THRESHOLD = 0.5
    else:
        BEST_THRESHOLD = 0.5

    # Try PyTorch .pth
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model = WaveletECGModel().to(device)
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            if verbose:
                print("Loaded PyTorch model successfully.")
            return {'model': model}
        except Exception as e:
            if verbose:
                print(f"Error loading PyTorch model: {e}")

    # Fallback: joblib
    if os.path.exists(joblib_path):
        try:
            model_dict = joblib.load(joblib_path)
            if verbose:
                print("Loaded joblib model successfully.")
            return model_dict
        except Exception as e:
            if verbose:
                print(f"Error loading joblib model: {e}")

    raise FileNotFoundError(f"No model found in {model_folder}")

def run_model(record, model_dict, verbose):

    global BEST_THRESHOLD
    device = get_device()

    if isinstance(model_dict, dict) and 'model' in model_dict:
        model = model_dict['model']
    else:
        model = model_dict  # fallback

    model.eval()
    model.to(device)

    try:
        signals, fields = load_signals(record)
    except Exception as e:
        if verbose:
            print(f"Error loading signals from {record}: {e}")
        # Return negative with very low probability
        return 0, 0.01

    # no augmentation in inference
    ecg = process_ecg_wavelet(signals, fields).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(ecg)
        prob = torch.sigmoid(logits).item()

    binary_output = int(prob >= BEST_THRESHOLD)
    probability_output = float(prob)

    if verbose:
        print(f"[run_model] record={record}, prob={prob:.4f}, threshold={BEST_THRESHOLD:.3f}, pred={binary_output}")

    return binary_output, probability_output
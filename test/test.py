from sklearn.metrics import confusion_matrix, roc_auc_score
import torch.nn as nn
import time
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import random
import os
import joblib

random.seed(33)
np.random.seed(33)
torch.manual_seed(33)
torch.cuda.manual_seed(33)
torch.cuda.manual_seed_all(33)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def read_fasta(file_name):
    sequences = {}
    current_id = ""
    current_sequence = ""
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_sequence:
                    sequences[current_id] = current_sequence
                current_id = line[1:]
                current_sequence = ""
            else:
                current_sequence += line.upper()
        if current_id and current_sequence:
            sequences[current_id] = current_sequence
    return sequences


def read_fasta_with_id_order(fasta_file, id_file):
    sequences = read_fasta(fasta_file)
    with open(id_file, 'r') as f:
        id_order = [line.strip() for line in f]
    return [sequences[pid] for pid in id_order]

class ProtGPT2FeatureExtractor:
    def __init__(self, model_name, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)
    def extract_features(self, sequences, batch_size=8, max_length=31):
        all_embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            pooled = last_hidden_state.mean(dim=1)
            all_embeddings.append(pooled.cpu())
        features = torch.cat(all_embeddings, dim=0).numpy()
        features = features[:, None, :]
        return features


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )
        out = self.fc_out(out)
        return out

class BiLSTM_attention_fusion_new(nn.Module):
    def __init__(self, input_size=433, hidden_size=128, num_layers=1, heads=8):
        super(BiLSTM_attention_fusion_new, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.multihead_attention = MultiHeadAttention(hidden_size * 2, heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2 * 2, 32)
        self.fc2 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        h_lstm = self.layer_norm1(h_lstm)
        attn_output = self.multihead_attention(h_lstm, h_lstm, h_lstm)
        attn_output = self.layer_norm2(attn_output)
        combined_features = torch.cat((h_lstm[:, -1, :], attn_output[:, -1, :]), dim=1)
        out = self.gelu(combined_features)
        out = self.fc1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out



def feature_selection(X_train, y_train, X_test, method='pca', n_dim=64, random_state=42, return_selector=False):
    is_3d = (X_train.ndim == 3)
    if is_3d:
        n, l, d = X_train.shape
        X_train_flat = X_train.reshape(-1, d)
        X_test_flat = X_test.reshape(-1, d)
    else:
        X_train_flat = X_train
        X_test_flat = X_test

    selector = None
    if method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=n_dim)
        selector.fit(X_train_flat, np.repeat(y_train, l) if is_3d else y_train)
        X_train_new = selector.transform(X_train_flat)
        X_test_new = selector.transform(X_test_flat)
    elif method == 'chi2':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train_nonneg = scaler.fit_transform(X_train_flat)
        X_test_nonneg = scaler.transform(X_test_flat)
        selector = SelectKBest(score_func=chi2, k=n_dim)
        selector.fit(X_train_nonneg, np.repeat(y_train, l) if is_3d else y_train)
        X_train_new = selector.transform(X_train_nonneg)
        X_test_new = selector.transform(X_test_nonneg)
    elif method == 'pca':
        selector = PCA(n_components=n_dim, svd_solver='randomized', random_state=random_state)
        selector.fit(X_train_flat)
        X_train_new = selector.transform(X_train_flat)
        X_test_new = selector.transform(X_test_flat)
    elif method == 'lgb':
        lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=random_state)
        lgbm.fit(X_train_flat, np.repeat(y_train, l) if is_3d else y_train)
        importances = lgbm.feature_importances_
        indices = np.argsort(importances)[::-1][:n_dim]
        selector = indices
        X_train_new = X_train_flat[:, indices]
        X_test_new = X_test_flat[:, indices]
    elif method == 'lasso':
        lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000, random_state=random_state)
        lasso.fit(X_train_flat, np.repeat(y_train, l) if is_3d else y_train)
        coef = np.abs(lasso.coef_).flatten()
        indices = np.argsort(coef)[::-1][:n_dim]
        selector = indices
        X_train_new = X_train_flat[:, indices]
        X_test_new = X_test_flat[:, indices]
    elif method == 'shap':
        import shap
        lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=random_state)
        lgbm.fit(X_train_flat, np.repeat(y_train, l) if is_3d else y_train)
        explainer = shap.TreeExplainer(lgbm)
        shap_values = explainer.shap_values(X_train_flat)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        indices = np.argsort(mean_abs_shap)[::-1][:n_dim]
        selector = indices
        X_train_new = X_train_flat[:, indices]
        X_test_new = X_test_flat[:, indices]
    else:
        raise ValueError("不支持的降维方式: {}".format(method))

    if is_3d:
        X_train_new = X_train_new.reshape(n, l, n_dim)
        X_test_new = X_test_new.reshape(X_test.shape[0], l, n_dim)

    if return_selector:
        return X_train_new, X_test_new, selector
    else:
        return X_train_new, X_test_new


def compute_metrics(y_true, y_pred_classes, y_pred_probs):
    cm = confusion_matrix(y_true, y_pred_classes, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = (0, 0, 0, 0)
        if y_true[0] == 0:
            tn = cm[0, 0]
        else:
            tp = cm[0, 0]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sn = tp / (tp + fn)
    sp = tn /(tn + fp)
    mcc = ((tp * tn) - (fp * fn)) /( ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)

    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred_probs)
    else:
        auc = 0.5
    return accuracy, sn, sp, mcc, auc


def main():
    selector = joblib.load("selector.joblib")
    device = torch.device("cpu")
    input_size = 300
    hidden_size = 256
    num_layers = 2
    model = BiLSTM_attention_fusion_new(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)
    model.load_state_dict(torch.load("best_model6.pth"))
    model.eval()
    feature_extractor = ProtGPT2FeatureExtractor(
        r"...\models--nferruz--ProtGPT2\snapshots\f71aa6cf063ad784ebd53881d11332fd098eaa58"
    )
    ce_files = [("", "")]
    for p_file, n_file in ce_files:
        ce_sequences_positive = read_fasta(p_file)
        ce_sequences_negative = read_fasta(n_file)
        ce_positive_sequences = list(ce_sequences_positive.values())
        ce_negative_sequences = list(ce_sequences_negative.values())
        ce_all_sequences = ce_positive_sequences + ce_negative_sequences
        ce_labels_positive = [1] * len(ce_positive_sequences)
        ce_labels_negative = [0] * len(ce_sequences_negative)
        ce_labels_all = ce_labels_positive + ce_labels_negative
        y_test = np.array(ce_labels_all)

        X_test = feature_extractor.extract_features(ce_all_sequences)
        is_3d = (X_test.ndim == 3)
        n_dim = input_size

        if is_3d:
            n, l, d = X_test.shape
            X_test_flat = X_test.reshape(-1, d)
        else:
            X_test_flat = X_test

        if hasattr(selector, "transform"):
            X_test_dim = selector.transform(X_test_flat)
        else:
            X_test_dim = X_test_flat[:, selector]

        if is_3d:
            X_test_dim = X_test_dim.reshape(n, l, n_dim)

        X_test_tensor = torch.tensor(X_test_dim, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(X_test_tensor)
            y_pred_classes = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_test_np = y_test_tensor.cpu().numpy()

        accuracy, sn, sp, mcc, auc = compute_metrics(y_test_np, y_pred_classes, y_pred_probs)
        print(f"Results for {p_file}/{n_file}:")
        print(f"# Final Accuracy (ACC): {accuracy:.4f}")
        print(f"# Final Sensitivity (SN): {sn:.4f}")
        print(f"# Final Specificity (SP): {sp:.4f}")
        print(f"# Final Matthews Correlation Coefficient (MCC): {mcc:.4f}")
        print(f"# Final AUC: {auc:.4f}" if auc is not None else "# Final AUC: 无法计算")
        print("-" * 50)

if __name__ == "__main__":
    main()
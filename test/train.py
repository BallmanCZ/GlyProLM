from sklearn.metrics import confusion_matrix
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import joblib
import random
import numpy as np
import torch
import time
from torch.utils.data import TensorDataset, DataLoader
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

random.seed(33)
np.random.seed(33)
torch.manual_seed(33)
torch.cuda.manual_seed(33)
torch.cuda.manual_seed_all(33)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def read_fasta(file_name):
    sequences = []
    current_sequence = ""
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_sequence:
                    sequences.append(current_sequence)
                current_sequence = ""
            else:
                current_sequence += line.upper()
        if current_sequence:
            sequences.append(current_sequence)
    return sequences


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

    positive_sequences = read_fasta('Train_Pos.txt')
    negative_sequences = read_fasta('Train_N(after undersampling).txt')
    all_sequences_resampled = positive_sequences + negative_sequences
    y_all = np.concatenate((
        np.ones(len(positive_sequences), dtype=int),
        np.zeros(len(negative_sequences), dtype=int)
    ))
    feature_extractor = ProtGPT2FeatureExtractor(
        r"...\models--nferruz--ProtGPT2\snapshots\f71aa6cf063ad784ebd53881d11332fd098eaa58"
    )
    X_all = feature_extractor.extract_features(all_sequences_resampled)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.15, random_state=42, stratify=y_all
    )

    dim_method = 'f_classif'
    n_dim = 300
    X_train_dim, X_val_dim, selector = feature_selection(
        X_train, y_train, X_val, method=dim_method, n_dim=n_dim, return_selector=True
    )
    joblib.dump(selector, "selector.joblib")

    device = torch.device("cpu")
    X_train_tensor = torch.tensor(X_train_dim, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val_dim, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    batch_size = 16
    num_epochs = 30
    learning_rate = 1e-5
    learning_rate_schedule = [10, 20, 25]
    learning_rate_drop_factor = 0.2
    hidden_size = 256
    num_layers = 2
    patience = 5

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)

    input_size = X_train_tensor.shape[2]
    model = BiLSTM_attention_fusion_new(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=learning_rate_schedule,
                                                     gamma=learning_rate_drop_factor)

    best_auc = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_tensor)
            y_val_probs = torch.softmax(logits_val, dim=1)[:, 1].cpu().numpy()
            y_val_np = y_val_tensor.cpu().numpy()
            try:
                auc = roc_auc_score(y_val_np, y_val_probs)
            except Exception:
                auc = 0.5

        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[EarlyStopping] No improvement after {patience} epochs, stop training!")
                break

if __name__ == "__main__":
    main()
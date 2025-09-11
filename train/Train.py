from itertools import product
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids
from collections import Counter
from transformers import EsmModel, EsmTokenizer, AutoTokenizer, AutoModel
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import time
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import  random
# 设置所有可能的随机种子
random.seed(33)
np.random.seed(33)
torch.manual_seed(33)
torch.cuda.manual_seed(33)
torch.cuda.manual_seed_all(33)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def UnderClusterCentroids(request, X, label):
    n_init = int(request.form.get('CCn_init'))
    voting = request.form.get('CC_voting')

    # 统计各类别样本数量
    label_counts = Counter(label.flatten())
    label_count_dict = dict(label_counts)
    print("原始样本数量:", label_count_dict)

    # 根据 request 中传入的参数构造采样策略
    for l, cnt in label_counts.items():
        num = int(request.form.get(f'ClusterCentroids{l}'))
        label_count_dict[l] = num
    print("采样策略:", label_count_dict)

    x_resampled, y_resampled = Cluster_Centroids(X, label, label_count_dict, n_init, voting)
    return x_resampled, y_resampled


def Cluster_Centroids(X, y, sampling_strategy, n_init, voting):

    y = y.ravel()
    under = ClusterCentroids(
        sampling_strategy=sampling_strategy,
        random_state=1,
        voting=voting,
        estimator=MiniBatchKMeans(n_init=n_init, random_state=1, batch_size=2048)
    )
    x_resampled, y_resampled = under.fit_resample(X, y)
    return x_resampled, y_resampled



def compute_dr(sequences, amino_acids="ACDEFGHIKLMNPQRSTVWYO", dmax=3):
    # 生成所有二肽的组合
    dipeptides = [''.join(pair) for pair in product(amino_acids, repeat=2)]
    feature_matrix = np.zeros((len(sequences), len(dipeptides)))

    for i, seq in enumerate(sequences):
        total_dipeptides = 0
        counts = Counter()

        # 计算 dmax 范围内的所有二肽
        for d in range(1, dmax + 1):
            for j in range(len(seq) - d):  # 注意修改，逐步增加间隔的长度
                dipeptide = seq[j] + seq[j + d]
                counts[dipeptide] += 1
                total_dipeptides += 1

        # 计算频率
        for j, dipeptide in enumerate(dipeptides):
            feature_matrix[i, j] = counts[dipeptide] / total_dipeptides if total_dipeptides > 0 else 0

    return feature_matrix




# 读取FASTA文件
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
                current_id = line[1:]  # 去掉 '>' 得到 ID，如 P1、N5
                current_sequence = ""
            else:
                current_sequence += line.upper()
        # 添加最后一个序列
        if current_id and current_sequence:
            sequences[current_id] = current_sequence
    return sequences





class ESMFeatureExtractor:
    def __init__(self, model_name, device='cpu'):
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)

    def extract_features(self, sequences, batch_size=64):
        all_embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=31
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1]  # 只取最后一层
            all_embeddings.append(last_hidden_state.cpu())
        return torch.cat(all_embeddings, dim=0).numpy()

from transformers import T5EncoderModel, AutoTokenizer

class Prot5FeatureExtractor:
    def __init__(self, model_name, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        if hasattr(self.model.config, "pad_token_id"):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)

    def extract_features(self, sequences, batch_size=16, max_length=31):
        all_embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            batch_seqs = [" ".join(list(seq)) for seq in batch_seqs]
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            # T5EncoderModel不需要token_type_ids
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "token_type_ids"}
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1]
            all_embeddings.append(last_hidden_state.cpu())
        return torch.cat(all_embeddings, dim=0).numpy()

import torch
from transformers import AutoTokenizer, AutoModel

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
            last_hidden_state = outputs.hidden_states[-1]  # [batch, seq_len, dim]
            # 均值池化到 [batch, dim]
            pooled = last_hidden_state.mean(dim=1)
            all_embeddings.append(pooled.cpu())
        features = torch.cat(all_embeddings, dim=0).numpy()  # [N, hidden_dim]
        # 补成 [N, 1, hidden_dim]
        features = features[:, None, :]
        return features  # shape: [N, 1, hidden_dim]



class CombinedFeatureExtractor:
    def __init__(self, esm_model_name, prot5_model_name, device='cpu'):
        self.esm_extractor = ESMFeatureExtractor(esm_model_name, device)
        self.prot5_extractor = Prot5FeatureExtractor(prot5_model_name, device)

    def extract_features(self, sequences):
        esm_feats = self.esm_extractor.extract_features(sequences)
        prot5_feats = self.prot5_extractor.extract_features(sequences)
        # 注意两边输出shape一致，否则需要池化
        esm_feats = torch.tensor(esm_feats)
        prot5_feats = torch.tensor(prot5_feats)
        # 直接拼接特征
        combined = torch.cat([esm_feats, prot5_feats], dim=-1)
        return combined.numpy()


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
        self.fc1 = nn.Linear(hidden_size * 2 * 2, 32)  # 由于拼接了两个特征，所以这里是hidden_size * 2 * 2
        self.fc2 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)  # 调整Dropout比例
        self.gelu = nn.GELU()  # 使用GELU激活函数
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        h_lstm = self.layer_norm1(h_lstm)  # 添加LayerNorm

        attn_output = self.multihead_attention(h_lstm, h_lstm, h_lstm)
        attn_output = self.layer_norm2(attn_output)  # 添加LayerNorm

        # 融合特征，拼接LSTM输出的最后一个时间步和多头注意力机制的输出
        combined_features = torch.cat((h_lstm[:, -1, :], attn_output[:, -1, :]), dim=1)

        out = self.gelu(combined_features)
        out = self.fc1(out)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out



def feature_selection(X_train, y_train, X_test, method='pca', n_dim=64, random_state=42):
    """
    三维输入（N, L, D）只降维最后一维D，输出依然三维。
    支持的method:
        'f_classif'  - F检验
        'chi2'       - 卡方
        'pca'        - PCA
        'lgb'        - LightGBM重要性
        'lasso'      - LASSO
        'shap'       - SHAP（基于lightgbm）
    """
    import numpy as np
    from sklearn.feature_selection import SelectKBest, f_classif, chi2
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb

    # 如果三维，合并N和L，做降维，最后reshape回去
    is_3d = (X_train.ndim == 3)
    if is_3d:
        n, l, d = X_train.shape
        X_train_flat = X_train.reshape(-1, d)    # (N*L, D)
        X_test_flat = X_test.reshape(-1, d)      # (N'*L, D)
    else:
        X_train_flat = X_train
        X_test_flat = X_test

    # 1. F检验
    if method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=n_dim)
        selector.fit(X_train_flat, np.repeat(y_train, l) if is_3d else y_train)
        X_train_new = selector.transform(X_train_flat)
        X_test_new = selector.transform(X_test_flat)
    # 2. 卡方 (先归一化到0~1)
    elif method == 'chi2':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train_nonneg = scaler.fit_transform(X_train_flat)
        X_test_nonneg = scaler.transform(X_test_flat)
        selector = SelectKBest(score_func=chi2, k=n_dim)
        selector.fit(X_train_nonneg, np.repeat(y_train, l) if is_3d else y_train)
        X_train_new = selector.transform(X_train_nonneg)
        X_test_new = selector.transform(X_test_nonneg)
    # 3. PCA
    elif method == 'pca':
        selector = PCA(n_components=n_dim, svd_solver='randomized', random_state=random_state)
        selector.fit(X_train_flat)
        X_train_new = selector.transform(X_train_flat)
        X_test_new = selector.transform(X_test_flat)
    # 4. LightGBM
    elif method == 'lgb':
        lgbm = lgb.LGBMClassifier(n_estimators=100, random_state=random_state)
        lgbm.fit(X_train_flat, np.repeat(y_train, l) if is_3d else y_train)
        importances = lgbm.feature_importances_
        indices = np.argsort(importances)[::-1][:n_dim]
        X_train_new = X_train_flat[:, indices]
        X_test_new = X_test_flat[:, indices]
    # 5. LASSO
    elif method == 'lasso':
        lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000, random_state=random_state)
        lasso.fit(X_train_flat, np.repeat(y_train, l) if is_3d else y_train)
        coef = np.abs(lasso.coef_).flatten()
        indices = np.argsort(coef)[::-1][:n_dim]
        X_train_new = X_train_flat[:, indices]
        X_test_new = X_test_flat[:, indices]
    # 6. SHAP
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
        X_train_new = X_train_flat[:, indices]
        X_test_new = X_test_flat[:, indices]
    else:
        raise ValueError("不支持的降维方式: {}".format(method))

    # reshape回三维
    if is_3d:
        X_train_new = X_train_new.reshape(n, l, n_dim)
        X_test_new = X_test_new.reshape(X_test.shape[0], l, n_dim)
    return X_train_new, X_test_new



# 计算各个评估指标
def compute_metrics(y_true, y_pred_classes, y_pred_probs):
    # 确保混淆矩阵包含所有标签
    cm = confusion_matrix(y_true, y_pred_classes, labels=[0, 1])

    # 如果混淆矩阵是2x2
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # 针对单一类别情况
        tn, fp, fn, tp = (0, 0, 0, 0)
        if y_true[0] == 0:  # 只有类别 0
            tn = cm[0, 0]
        else:  # 只有类别 NCR+ENN
            tp = cm[0, 0]

    # 计算指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sn = tp / (tp + fn) # 敏感性
    sp = tn /(tn + fp) # 特异性
    mcc = ((tp * tn) - (fp * fn)) /( ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
    # 马修斯相关系数

    # AUC 计算
    if len(set(y_true)) > 1:  # 如果y_true中有两个类别
        auc = roc_auc_score(y_true, y_pred_probs)
    else:
        auc = 0.5  # 单类别时AUC无意义

    return accuracy, sn, sp, mcc, auc

def fasta_file_to_sequence_list(file_path):
    """
    读取FASTA文件，返回序列组成的列表（顺序与文件一致）
    """
    sequences = []
    current_seq = ""
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append(current_seq)
                current_seq = ""
            else:
                current_seq += line.upper()
        if current_seq:
            sequences.append(current_seq)
    return sequences

# 主函数
def main():
    # 直接调用就能得到你要的格式
    positive_sequences_resampled = fasta_file_to_sequence_list('Train_Pos.txt')
    negative_sequences_resampled = fasta_file_to_sequence_list('Train_N(实际1).txt')

    print("正样本数量：", len(positive_sequences_resampled))
    print("负样本数量：", len(negative_sequences_resampled))

    # ========== Step 3. 十折交叉验证 + 降维 + BiLSTM分类 ==========

    device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # 1. 十折分割正负样本，保持每折正负比例一致
    positive_samples = np.random.permutation(positive_sequences_resampled).tolist()
    negative_samples = np.random.permutation(negative_sequences_resampled).tolist()
    positive_splits = [list(split) for split in np.array_split(positive_samples, 10)]
    negative_splits = [list(split) for split in np.array_split(negative_samples, 10)]
    combined_splits = []
    for i in range(10):
        combined_sequences = positive_splits[i] + negative_splits[i]
        combined_labels = [1] * len(positive_splits[i]) + [0] * len(negative_splits[i])
        combined_splits.append((combined_sequences, combined_labels))

    # 2. 六种降维方式
    # dim_methods = ['f_classif', 'chi2', 'pca', 'lgb', 'lasso', 'shap']
    dim_methods = ['f_classif']
    n_dim = 300  # 降维后维度

    from sklearn.model_selection import KFold, train_test_split
    from torch.utils.data import TensorDataset, DataLoader, Subset

    for method in dim_methods:
        print(f"\n========== 降维方法: {method}, 目标维度: {n_dim} ==========")
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        metrics = []

        for fold, (train_index, test_index) in enumerate(kf.split(combined_splits), 1):
            print(f"[INFO] Processing Fold {fold}...")

            # 3. 构建当前折的训练和测试集（序列和标签）
            X_train, y_train_fold = [], []
            X_test, y_test_fold = [], []
            for i in train_index:
                X_train.extend(combined_splits[i][0])
                y_train_fold.extend(combined_splits[i][1])
            for i in test_index:
                X_test.extend(combined_splits[i][0])
                y_test_fold.extend(combined_splits[i][1])
            print(f"[INFO] Fold {fold} 样本数: Train={len(X_train)}, Test={len(X_test)}")

            # 4. ProtGPT2特征提取（每一折分别提取，确保信息不泄露）
            feature_extractor = ProtGPT2FeatureExtractor(
                r"C:\Users\21353\.cache\huggingface\hub\models--nferruz--ProtGPT2\snapshots\f71aa6cf063ad784ebd53881d11332fd098eaa58"
                # 你的本地路径，或直接用 "nferruz/ProtGPT2"
            )
            extracted_features_train = feature_extractor.extract_features(X_train, batch_size=4)
            extracted_features_test = feature_extractor.extract_features(X_test, batch_size=4)

            print("[DEBUG] extracted_features_train shape:", extracted_features_train.shape)
            print("[DEBUG] extracted_features_test shape:", extracted_features_test.shape)

            # 5. 降维（每一折都要做）
            try:
                X_train_dim, X_test_dim = feature_selection(
                    extracted_features_train, y_train_fold, extracted_features_test, method=method, n_dim=n_dim
                )
            except Exception as e:
                print(f"[ERROR] 方法 {method} 在 Fold {fold} 发生异常: {e}")
                continue
            print(f"降维后X_train shape: {X_train_dim.shape}, 降维后X_test shape: {X_test_dim.shape}")

            # 6. 转为Tensor
            X_train_tensor = torch.tensor(X_train_dim, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train_fold, dtype=torch.long).to(device)
            X_test_tensor = torch.tensor(X_test_dim, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test_fold, dtype=torch.long).to(device)

            # 6.5 从训练折里再划一份验证集 —— 只用于早停与保存最优模型（关键修复点）
            #    这里按 15% 作为验证集比例，你可按需调整
            tr_indices = np.arange(len(X_train_tensor))
            # 注意：stratify 需要在 CPU 上的 numpy 数组
            y_train_np_for_split = y_train_tensor.detach().cpu().numpy()
            tr_sub_idx, val_idx = train_test_split(
                tr_indices, test_size=0.15, random_state=42, stratify=y_train_np_for_split
            )

            # 构建 DataLoader：训练集与验证集
            full_train_ds = TensorDataset(X_train_tensor, y_train_tensor)
            train_ds = Subset(full_train_ds, tr_sub_idx)
            val_ds = Subset(full_train_ds, val_idx)

            # 7. DataLoader和训练参数（保持原设置）
            batch_size = 16
            num_epochs = 30
            learning_rate = 1e-5
            learning_rate_schedule = [10, 20, 25]  # [7]
            learning_rate_drop_factor = 0.2
            hidden_size = 256
            num_layers = 2
            patience = 5

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            # 注意：如果 X_train_tensor 是 [N, L, F]，下面的 input_size 维度索引与原来一致
            input_size = X_train_tensor.shape[2]
            model = BiLSTM_attention_fusion_new(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=learning_rate_schedule, gamma=learning_rate_drop_factor
            )

            best_auc = 0
            patience_counter = 0

            # 8. 训练主循环
            start_time = time.time()
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                scheduler.step()
                avg_loss = epoch_loss / len(train_loader)

                # —— 关键修复：EarlyStopping 改为“监控验证集 AUC”，不再用测试集 —— #
                model.eval()
                val_probs_list, val_labels_list = [], []
                with torch.no_grad():
                    for vX, vY in val_loader:
                        logits_val = model(vX)
                        probs_val = torch.softmax(logits_val, dim=1)[:, 1].cpu().numpy()
                        val_probs_list.append(probs_val)
                        val_labels_list.append(vY.cpu().numpy())
                y_val_probs = np.concatenate(val_probs_list) if len(val_probs_list) else np.array([])
                y_val_np = np.concatenate(val_labels_list) if len(val_labels_list) else np.array([])

                if y_val_probs.size > 0:
                    try:
                        auc = roc_auc_score(y_val_np, y_val_probs)
                    except Exception:
                        auc = 0.5
                else:
                    # 极端情况下（验证集非常小）保证流程可运行
                    auc = 0.5

                print(f"Fold {fold} - Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f} - Val AUC: {auc:.4f}")

                if auc > best_auc:
                    best_auc = auc
                    patience_counter = 0
                    torch.save(model.state_dict(), f"best_model_fold{fold}_{method}.pt")
                else:
                    patience_counter += 1
                    # 保持你的原逻辑（不修改 off-by-one）
                    if patience_counter > patience:
                        print(f"[EarlyStopping] No improvement after {patience} epochs, stop training!")
                        break

            training_time = time.time() - start_time
            print(f"[INFO] Fold {fold} Training completed in {training_time:.2f} seconds.")

            # 9. 载入最优模型做评估 —— 只在测试集上评估一次
            model.load_state_dict(torch.load(f"best_model_fold{fold}_{method}.pt"))
            model.eval()
            with torch.no_grad():
                logits = model(X_test_tensor)
                y_pred_classes = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                y_test_np = y_test_tensor.cpu().numpy()
            accuracy, sn, sp, mcc, auc = compute_metrics(y_test_np, y_pred_classes, y_pred_probs)
            metrics.append((accuracy, sn, sp, mcc, auc))

            print(f"Fold {fold} Results:")
            print(f"  Accuracy (ACC): {accuracy:.4f}")
            print(f"  Sensitivity (SN): {sn:.4f}")
            print(f"  Specificity (SP): {sp:.4f}")
            print(f"  Matthews Correlation Coefficient (MCC): {mcc:.4f}")
            print(f"  AUC: {auc:.4f}" if auc is not None else "  AUC: 无法计算")
            print("-" * 50)

        # 10. 汇总各折结果
        accs = [m[0] for m in metrics]
        sns = [m[1] for m in metrics]
        sps = [m[2] for m in metrics]
        mccs = [m[3] for m in metrics]
        aucs = [m[4] for m in metrics if m[4] is not None]
        print(f"【{method}】# Average Accuracy (ACC): {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"【{method}】# Average Sensitivity (SN): {np.mean(sns):.4f} ± {np.std(sns):.4f}")
        print(f"【{method}】# Average Specificity (SP): {np.mean(sps):.4f} ± {np.std(sps):.4f}")
        print(f"【{method}】# Average Matthews Correlation Coefficient (MCC): {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")
        if aucs:
            print(f"【{method}】# Average AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        else:
            print(f"【{method}】# Average AUC: 无法计算")


if __name__ == "__main__":
    main()

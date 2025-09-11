from itertools import product
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids
from collections import Counter
import numpy as np
import torch
import random
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
    dipeptides = [''.join(pair) for pair in product(amino_acids, repeat=2)]
    feature_matrix = np.zeros((len(sequences), len(dipeptides)))
    for i, seq in enumerate(sequences):
        total_dipeptides = 0
        counts = Counter()
        for d in range(1, dmax + 1):
            for j in range(len(seq) - d):
                dipeptide = seq[j] + seq[j + d]
                counts[dipeptide] += 1
                total_dipeptides += 1
        for j, dipeptide in enumerate(dipeptides):
            feature_matrix[i, j] = counts[dipeptide] / total_dipeptides if total_dipeptides > 0 else 0
    return feature_matrix

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

def write_fasta(sequences_dict, ids, file_name):
    with open(file_name, 'w') as f:
        for id_ in ids:
            f.write(f">{id_}\n{sequences_dict[id_]}\n")

def get_unique_nearest_ids(cluster_centers, neg_features, neg_ids, target_count):
    """
    对于每个聚类中心，找到最近的且未被选中过的负样本，保证无重复且采样数量等于正样本数量
    """
    used_indices = set()
    selected_ids = []
    for center in cluster_centers:
        dists = np.linalg.norm(neg_features - center, axis=1)
        sorted_indices = np.argsort(dists)
        for idx in sorted_indices:
            if idx not in used_indices:
                used_indices.add(idx)
                selected_ids.append(neg_ids[idx])
                break
        if len(selected_ids) == target_count:
            break
    # 补充（如果聚类中心数少于目标数量，理论极少发生）
    if len(selected_ids) < target_count:
        for idx, nid in enumerate(neg_ids):
            if idx not in used_indices:
                selected_ids.append(nid)
            if len(selected_ids) == target_count:
                break
    return selected_ids

def main():
    # ========== Step 1. 读取训练集并处理 ==========
    sequences_positive = read_fasta('Ce_P.txt')
    print(f"[INFO] 正样本数量: {len(sequences_positive)}")
    positive_ids = list(sequences_positive.keys())
    positive_seqs = list(sequences_positive.values())
    encoded_positive = compute_dr(positive_seqs)
    print(f"[INFO] 正样本编码完成，shape: {encoded_positive.shape}")
    labels_positive = np.ones(len(encoded_positive), dtype=int)
    positive_ids_col = np.array(positive_ids).reshape(-1, 1)
    positive_data = np.hstack((encoded_positive, positive_ids_col))
    print(f"[INFO] 正样本数据（含编号） shape: {positive_data.shape}")

    sequences_negative = read_fasta('Ce_N.txt')
    print(f"[INFO] 负样本数量: {len(sequences_negative)}")
    negative_ids = list(sequences_negative.keys())
    negative_seqs = list(sequences_negative.values())
    encoded_negative = compute_dr(negative_seqs)
    print(f"[INFO] 负样本编码完成，shape: {encoded_negative.shape}")
    labels_negative = np.zeros(len(encoded_negative), dtype=int)
    negative_ids_col = np.array(negative_ids).reshape(-1, 1)
    negative_data = np.hstack((encoded_negative, negative_ids_col))
    print(f"[INFO] 负样本数据（含编号） shape: {negative_data.shape}")

    full_data = np.vstack((positive_data, negative_data))
    X_all = full_data[:, :-1].astype(float)
    y_all = np.concatenate((labels_positive, labels_negative))
    print(f"[INFO] 合并后的数据 shape: {X_all.shape}, 标签 shape: {y_all.shape}")

    # ========== Step 2. 负样本下采样 ==========
    negative_target_count = len(positive_data)
    print(f"[INFO] 启动 ClusterCentroids 下采样，目标负样本数: {negative_target_count}")

    # 伪造request对象
    class DummyRequest:
        form = {
            'CCn_init': '10',
            'CC_voting': 'hard',
            'ClusterCentroids0': str(negative_target_count),
            'ClusterCentroids1': str(len(positive_seqs))
        }
    dummy_request = DummyRequest()

    X_resampled, y_resampled = UnderClusterCentroids(dummy_request, X_all, y_all)
    print(f"[INFO] 下采样完成，shape: {X_resampled.shape}, 标签 shape: {y_resampled.shape}")

    # 找出负样本部分的特征、id
    neg_mask = y_all == 0
    neg_features = X_all[neg_mask]
    neg_ids = np.array(negative_ids)

    # 保证无重复id的采样
    unique_negative_resampled_ids = get_unique_nearest_ids(
        cluster_centers=X_resampled[y_resampled == 0],
        neg_features=neg_features,
        neg_ids=neg_ids,
        target_count=negative_target_count
    )

    # 提取下采样后正负样本的原始序列
    positive_sequences_resampled = [sequences_positive[pid] for pid in positive_ids]
    negative_sequences_resampled = [sequences_negative[nid] for nid in unique_negative_resampled_ids]

    print("前三条正样本id和序列：")
    for idx in range(3):
        pid = positive_ids[idx]
        seq = sequences_positive[pid]
        print(f"正样本ID: {pid}\t序列: {seq}")

    print("前三条负样本id和序列：")
    for idx in range(3):
        nid = unique_negative_resampled_ids[idx]
        seq = sequences_negative[nid]
        print(f"负样本ID: {nid}\t序列: {seq}")

    # 合并下采样后的正负样本序列，用于特征提取
    all_sequences_resampled = positive_sequences_resampled + negative_sequences_resampled
    y_train = np.concatenate((np.ones(len(positive_sequences_resampled)), np.zeros(len(negative_sequences_resampled))))

    # ========== 输出下采样后的负样本为FASTA ==========
    output_filename = "Ce_N(实际1).txt"
    write_fasta(sequences_negative, unique_negative_resampled_ids, output_filename)
    print(f"[INFO] 已输出下采样后的负样本FASTA文件: {output_filename}")

if __name__ == "__main__":
    main()

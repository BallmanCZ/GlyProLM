import numpy as np
import random

random.seed(40)
np.random.seed(40)

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

# 主函数
def main():
    # 读取原始数据
    positive_sequences = read_fasta('Train_P3969.txt')
    negative_sequences = read_fasta('Train_N82270.txt')

    random.shuffle(positive_sequences)
    random.shuffle(negative_sequences)

    num_negative_selected = int(0.1 * len(negative_sequences))
    selected_negative_for_Ce = random.sample(negative_sequences, num_negative_selected)
    with open('Ce_N1.txt', 'w') as f:
        for idx, seq in enumerate(selected_negative_for_Ce):
            f.write(f">N{idx + 1}\n" + seq + "\n")

    num_positive_selected = int(0.1 * len(positive_sequences))
    selected_positive_for_Ce = random.sample(positive_sequences, num_positive_selected)
    with open('Ce_P1.txt', 'w') as f:
        for idx, seq in enumerate(selected_positive_for_Ce):
            f.write(f">P{idx + 1}\n" + seq + "\n")

    remaining_positive = [seq for seq in positive_sequences if seq not in selected_positive_for_Ce]
    with open('Train_P1.txt', 'w') as f:
        for idx, seq in enumerate(remaining_positive):
            f.write(f">P{idx + 1}\n" + seq + "\n")

    remaining_negative = [seq for seq in negative_sequences if seq not in selected_negative_for_Ce]
    with open('Train_N1.txt', 'w') as f:
        for idx, seq in enumerate(remaining_negative):
            f.write(f">N{idx + 1}\n" + seq + "\n")

if __name__ == "__main__":
    main()



# def read_fasta_with_id(filename):
#     pairs = []
#     curr_id, curr_seq = None, ""
#     with open(filename) as f:
#         for line in f:
#             line = line.strip()
#             if line.startswith(">"):
#                 if curr_id is not None:
#                     pairs.append((curr_id, curr_seq))
#                 curr_id = line[1:]
#                 curr_seq = ""
#             else:
#                 curr_seq += line.upper()
#         if curr_id is not None:
#             pairs.append((curr_id, curr_seq))
#     return pairs
#
# def map_sample_to_all(sample_pairs, seq2ids, used_ids, out_file):
#     seen_seq = set()
#     duplicates_in_sample = set()
#     lines = []
#     for new_id, seq in sample_pairs:
#         if seq in seen_seq:
#             duplicates_in_sample.add(seq)
#         seen_seq.add(seq)
#         if seq in seq2ids:
#             for old_id in seq2ids[seq]:
#                 if old_id not in used_ids:
#                     lines.append(f"{new_id}\t{old_id}")
#                     used_ids.add(old_id)
#                     break
#             else:
#                 lines.append(f"{new_id}\t[未找到可用原编号]")
#         else:
#             lines.append(f"{new_id}\t[原文件未找到该序列]")
#
#     with open(out_file, "w") as out:
#         out.write("new_id\told_id\n")
#         for l in lines:
#             out.write(l + "\n")
#     if duplicates_in_sample:
#         print(f"注意：{out_file} 中有重复序列：")
#         for s in duplicates_in_sample:
#             print(f"重复序列: {s}")
#
# def main():
#     all_file = "Train_N82270.txt"
#     sample1_file = "Train_Neg.txt"
#     sample2_file = "Ce_N.txt"
#     output1 = "Train_Neg_match_sample1.txt"
#     output2 = "Ce_N_Trainmatch_sample2.txt"
#
#     # 读取全集和两个子集
#     all_pairs = read_fasta_with_id(all_file)
#     sample1_pairs = read_fasta_with_id(sample1_file)
#     sample2_pairs = read_fasta_with_id(sample2_file)
#
#     # 建全集序列到所有编号的映射
#     from collections import defaultdict
#     seq2ids = defaultdict(list)
#     for old_id, seq in all_pairs:
#         seq2ids[seq].append(old_id)
#
#     used_ids = set()
#     # 1. 先处理sample1
#     map_sample_to_all(sample1_pairs, seq2ids, used_ids, output1)
#     # 2. 再处理sample2，在剩余全集里继续找
#     map_sample_to_all(sample2_pairs, seq2ids, used_ids, output2)
#
#     print(f"匹配完成！结果写入：{output1} 和 {output2}")
#
# if __name__ == "__main__":
#     main()

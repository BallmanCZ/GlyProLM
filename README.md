# GlyProLM
GlyProLM uses Distance Residue (DR, dmax=3) features to represent protein sequences and applies Cluster Centroids Undersampling (CCU) to address the imbalance between positive and negative samples, ensuring more representative training data.

On this basis, the protein pre-trained model ProtGPT2 is introduced to generate 1280-dimensional semantic embeddings, capturing the contextual information of protein sequences. A univariate F-test is then applied to reduce the feature dimension to 300, minimizing redundancy and improving generalization.

For classification, a deep learning framework combining Bidirectional Long Short-Term Memory (BiLSTM) networks with Multi-Head Attention (MHA) is constructed to capture both bidirectional dependencies and critical site features.

Results from ten-fold cross-validation and independent testing show that GlyProLM achieves superior sensitivity (SN) compared to existing methods, improving the recall of lysine glycation sites. At the same time, it maintains strong performance in specificity (SP), Matthews correlation coefficient (MCC), accuracy (ACC), and AUC, demonstrating its overall effectiveness in identifying lysine post-translational modification sites.

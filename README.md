# GlyProLM
GlyProLM uses Distance Residue features to represent protein sequences and applies Cluster Centroids Undersampling (CCU) to address the imbalance between positive and negative samples, ensuring more representative training data.

On this basis, the protein pre-trained model ProtGPT2 is introduced to generate 1280-dimensional semantic embeddings, capturing the contextual information of protein sequences. A univariate F-test is then applied to reduce the feature dimension to 300, minimizing redundancy and improving generalization.

For classification, a deep learning framework combining Bidirectional Long Short-Term Memory (BiLSTM) networks with Multi-Head Attention (MHA) is constructed to capture both bidirectional dependencies and critical site features.

Results from ten-fold cross-validation and independent testing show that GlyProLM achieves superior sensitivity (SN) compared to existing methods, improving the recall of lysine glycation sites. At the same time, it maintains strong performance in specificity (SP), Matthews correlation coefficient (MCC), accuracy (ACC), and AUC, demonstrating its overall effectiveness in identifying lysine post-translational modification sites.
# How to run
 (1) data  
This folder contains ten partitions of independent test sets with positive and negative samples, which are used for independent evaluation and validation of the model.  

 (2) original data partitioning  
This folder stores the partitioning results of the original dataset. Specifically, the positive and negative samples are divided into training and testing sets at a 9:1 ratio, ensuring a scientific and reproducible data split.  

 (3) test  
This folder includes the code for independent testing. It is mainly used to load trained models and read the corresponding datasets from the *data* folder to perform model evaluation and performance testing.  

 (4) train  
This folder contains the core code for model training. It implements balancing of positive and negative training sets, as well as the ten-fold cross-validation process, to enhance the model’s generalization ability and robustness.  

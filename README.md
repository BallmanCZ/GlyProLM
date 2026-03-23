# GlyProLM

## Overview
GlyProLM is a computational framework for identifying lysine glycation sites in proteins. It integrates sequence-derived features, protein language model embeddings, feature selection, and deep learning to improve the prediction of lysine post-translational modification sites.

Specifically, GlyProLM first uses distance residue-based features to represent protein sequences and applies Cluster Centroids Undersampling (CCU) to alleviate the imbalance between positive and negative samples, thereby improving the representativeness of the training data. On this basis, the protein pre-trained language model ProtGPT2 is employed to generate 1280-dimensional semantic embeddings that capture contextual information from protein sequences. A univariate F-test is then applied to reduce the feature dimension to 300, which helps decrease redundancy and improve model generalization.

For classification, GlyProLM adopts a deep learning architecture combining Bidirectional Long Short-Term Memory (BiLSTM) networks with Multi-Head Attention (MHA), enabling the model to capture both bidirectional sequence dependencies and critical site-related information.

Results from ten-fold cross-validation and independent testing demonstrate that GlyProLM achieves superior sensitivity (SN) compared with existing methods, while maintaining strong performance in specificity (SP), Matthews correlation coefficient (MCC), accuracy (ACC), and AUC. These results indicate that GlyProLM is an effective framework for lysine glycation site prediction.

---

## Main Features
- Distance residue-based feature representation of protein sequences
- Cluster Centroids Undersampling (CCU) to address class imbalance
- ProtGPT2-based 1280-dimensional semantic embeddings
- Univariate F-test for feature selection (300 dimensions retained)
- BiLSTM + Multi-Head Attention classifier
- Evaluation through ten-fold cross-validation and independent testing

---

## Repository Structure

### 1. `data/`
This folder contains the independent test datasets, including positive and negative samples, which are used for model evaluation and validation.

### 2. `original data partitioning/`
This folder stores the partitioning results of the original dataset. Specifically, the positive and negative samples are divided into training and testing sets at a 9:1 ratio to ensure a reproducible and scientifically reasonable data split.

### 3. `test/`
This folder contains the code and files for independent testing and inference. It includes the trained model weights and the related files required for evaluation.

Main files in this folder include:
- `best_model.pth`: the final trained model checkpoint used in this study
- `selector.joblib`: the feature selector required for inference
- `test.py`: the script for independent testing and evaluation

### 4. `train/`
This folder contains the core code for model training, including training-set balancing and ten-fold cross-validation, in order to improve the model’s robustness and generalization ability.

---

## Trained Model Weights
The trained model weights used in this study are publicly available in this repository.

The following files are provided for direct use:
- `test/best_model.pth`: final trained checkpoint of GlyProLM
- `test/selector.joblib`: feature selector used in the prediction pipeline

The file `best_model.pth` corresponds to the final trained model used for the results reported in the manuscript.

---

## Dependencies
GlyProLM is implemented in Python. The recommended environment includes the following packages:

- Python 3.8+
- PyTorch
- Transformers
- NumPy
- scikit-learn
- joblib

Dependencies can be installed manually, for example:

```bash
pip install torch transformers numpy scikit-learn joblib
```

If a `requirements.txt` file is provided, the dependencies can also be installed using:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Independent Testing / Inference
To perform independent testing using the released trained model:

```bash
cd test
python test.py
```

This script uses the trained checkpoint (`best_model.pth`) together with the feature selector (`selector.joblib`) and reads the corresponding dataset files for evaluation.

---

## Reproducibility and Availability
To improve the reproducibility and usability of this work, the source code, datasets, and trained model weights are all publicly available in this repository. The repository includes the trained checkpoint, the associated inference files, and the testing code required for reproducing the prediction pipeline without retraining the model from scratch.

---

## Notes
- Please ensure that the model-loading path in the testing script is consistent with the released checkpoint file name (`best_model.pth`).
- Please ensure that all required dependencies are installed before running the testing code.
- If additional path configuration is needed in your local environment, please update the script accordingly.

---

## Citation
If you find this work useful, please cite the corresponding paper of GlyProLM.

---

## Conclusion
GlyProLM provides an effective framework for lysine glycation site prediction by integrating sequence-derived features, protein language model embeddings, feature selection, and deep learning. By making the code, datasets, and trained model weights publicly available, this repository aims to support transparent, reproducible, and convenient reuse by the research community.

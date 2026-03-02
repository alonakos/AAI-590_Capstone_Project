# AAI-590_Capstone_Project

## Automated Pulsar Candidate Classification
This project implements a supervised machine learning pipeline to classify pulsar candidates from the HTRU2 dataset. The objective is to distinguish true radio pulsars from noise using statistical features derived from radio survey data. The system simulates automated candidate triage in large-scale astronomical surveys.

## Dataset
The HTRU2 dataset contains 17,898 labeled examples with eight numerical features per candidate and one binary class label (pulsar / non-pulsar).

Feature groups:
- **Integrated Pulse Profile**: mean, standard deviation, skewness, kurtosis  
- **DM–SNR Curve**: mean, standard deviation, skewness, kurtosis  

The dataset reflects realistic survey conditions, including significant class imbalance.

## Models
### Baseline Models
- Logistic Regression  
- Random Forest  

### Deep Learning Model
- Feed-forward Multilayer Perceptron (MLP)

## Evaluation
Models are evaluated using metrics appropriate for imbalanced binary classification:
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

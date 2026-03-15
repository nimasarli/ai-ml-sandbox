**Breast Cancer Classification** 

Using scikit-learn, do classification of breast cancer based on biopsy features.
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
# 0->malignant and 1->benign
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='cancer_type')
```

Split the data into training and testing, and train different classifiers to classify into either benign or malignant. Compute test metrics on the test dataset (accuracy, confusion matrix, recall and precision, etc.). Compare different classifiers of your choice -- including for example, logistic regression, bag of decision trees, random forest and kNN.

**Options/probing further:**

1. Display labeled point clouds in 2D, probe dimensionality reduction methods like PCA, or t-SNE or UMAP for visualization.
2. Treat the problem as an unsupervised learning problem and do clustering using KMeans, mean-shift or other methods, use the same visualization approach as in 1 to show clusters or simply plot the first and second feature.

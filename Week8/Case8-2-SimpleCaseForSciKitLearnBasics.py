""""A simple case
https://www.datacamp.com/tutorial/machine-learning-python

"""
import pandas as pd
from sklearn.datasets import load_wine


"""
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
load_wine
sklearn.datasets.load_wine(*, return_X_y=False, as_frame=False)[source]
Load and return the wine dataset (classification).

https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset

"""
wine_data = load_wine()


wine_data_return_X_y = load_wine( return_X_y= True)
"""(data, target)tuple if return_X_y is True
A tuple of two ndarrays by default. The first contains a 2D array of shape (178, 13) with each row representing one sample and each column representing the features. The second array of shape (178,) contains the target samples."""
print("wine_data_return_X_y :    ", wine_data_return_X_y)
print("wine_data_return_X_y[0] :    ", wine_data_return_X_y[0])
print("wine_data_return_X_y[1] :    ", wine_data_return_X_y[1])

wine_data_as_frame = load_wine(as_frame=True)
print("wine_data_as_frame", wine_data_as_frame )


# Convert data to pandas dataframe
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

print("wine_df - dataFrame: ", wine_df)

# Add the target label
wine_df["target"] = wine_data.target


# Take a preview
print("wine_df.head() : ", wine_df.head())


print(" wine_df.info() ", wine_df.info() )

print(" wine_df.describe()  ", wine_df.describe()  )


print("wine_df.tail()" , wine_df.tail() )


from sklearn.preprocessing import StandardScaler

# Split data into features and label 
X = wine_df[wine_data.feature_names].copy()
y = wine_df["target"].copy() 

print("X:" , X)
print("y:" , y)

""""
https://scikit-learn.org/0.24/modules/generated/sklearn.preprocessing.StandardScaler.html

Standardize features by removing the mean and scaling to unit variance

The standard score of a sample x is calculated as:

z = (x - u) / s

where u is the mean of the training samples or zero if with_mean=False, and s is the standard deviation of the training samples or one if with_std=False.

Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using transform.

Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance)."""
# Instantiate scaler and fit on features
scaler = StandardScaler()
scaler.fit(X)

# Transform features
X_scaled = scaler.transform(X.values)

# View first instance
print(X_scaled[0])


from sklearn.model_selection import train_test_split

#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
""""train_test_split
sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)[source]
Split arrays or matrices into random train and test subsets.

Quick utility that wraps input validation, next(ShuffleSplit().split(X, y)), and application to input data into a single call for splitting (and optionally subsampling) data into a one-liner."""
# Split data into train and test
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled,
                                                                  y,
                                                             train_size=.7,
                                                           random_state=25)

# Check the splits are correct
print(f"Train size: {round(len(X_train_scaled) / len(X) * 100)}% \n\
Test size: {round(len(X_test_scaled) / len(X) * 100)}%")

"""
Train size: 70% 
Test size: 30%"""


""""
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

class sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='deprecated', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)[source]
Logistic Regression (aka logit, MaxEnt) classifier.

This class implements regularized logistic regression using the ‘liblinear’ library, ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ solvers. Note that regularization is applied by default. It can handle both dense and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit floats for optimal performance; any other input format will be converted (and copied).

The ‘newton-cg’, ‘sag’, and ‘lbfgs’ solvers support only L2 regularization with primal formulation, or no regularization. The ‘liblinear’ solver supports both L1 and L2 regularization, with a dual formulation only for the L2 penalty. The Elastic-Net regularization is only supported by the ‘saga’ solver.

================================
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

SVC
class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)[source]
C-Support Vector Classification.

The implementation is based on libsvm. The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples. For large datasets consider using LinearSVC or SGDClassifier instead, possibly after a Nystroem transformer or other Kernel Approximation.

The multiclass support is handled according to a one-vs-one scheme.

For details on the precise mathematical formulation of the provided kernel functions and how gamma, coef0 and degree affect each other, see the corresponding section in the narrative documentation: Kernel functions.

========================
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

DecisionTreeClassifier
class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0, monotonic_cst=None)[source]
A decision tree classifier.

"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Instnatiating the models 
logistic_regression = LogisticRegression()
svm = SVC()
tree = DecisionTreeClassifier()

# Training the models 
logistic_regression.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
tree.fit(X_train_scaled, y_train)

# Making predictions with each model
log_reg_preds = logistic_regression.predict(X_test_scaled)
svm_preds = svm.predict(X_test_scaled)
tree_preds = tree.predict(X_test_scaled)

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
""""
classification_report
sklearn.metrics.classification_report(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')[source]
Build a text report showing the main classification metrics.
"""
from sklearn.metrics import classification_report

# Store model predictions in a dictionary
# this makes it's easier to iterate through each model
# and print the results. 
model_preds = {
    "Logistic Regression": log_reg_preds,
    "Support Vector Machine": svm_preds,
    "Decision Tree": tree_preds
}

for model, preds in model_preds.items():
    print(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")



""""
https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report

3.4.4.9. Precision, recall and F-measures
Intuitively, precision is the ability of the classifier not to label as positive a sample that is negative, and recall is the ability of the classifier to find all the positive samples.

The F-measure (
 and 
 measures) can be interpreted as a weighted harmonic mean of the precision and recall. A 
 measure reaches its best value at 1 and its worst score at 0. With 
, 
 and 
 are equivalent, and the recall and the precision are equally important.

The precision_recall_curve computes a precision-recall curve from the ground truth label and a score given by the classifier by varying a decision threshold.

The average_precision_score function computes the average precision (AP) from prediction scores. The value is between 0 and 1 and higher is better. AP is defined as

 
where 
 and 
 are the precision and recall at the nth threshold. With random predictions, the AP is the fraction of positive samples.

"""
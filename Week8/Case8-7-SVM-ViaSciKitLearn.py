"""
https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/


Data: https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt

"""


"""
Let's import the data into a pandas dataframe structure, and take a look at its first five rows with the head() method.

Notice that the data is saved in a txt (text) file format, separated by commas, and it is without a header. We can reconstruct it as a table by reading it as a csv, specifying the separator as a comma, and adding the column names with the names argument.

Let's follow those three steps at once, and then look at the first five rows of the data:

"""
import pandas as pd
import matplotlib.pyplot as plt

data_link = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
col_names = ["variance", "skewness", "curtosis", "entropy", "class"]

bankdata = pd.read_csv(data_link, names=col_names, sep=",", header=None)
bankdata.head()


"""
The fifth variable was the class variable, which probably has 0 and 1 values, that say if the note was real or forged.

We can check if the fifth column contain zeros and ones with Pandas' unique() method:
"""

print(bankdata['class'].unique())

"""
We can also see how many records, or images we have, by looking at the number of rows in the data via the shape property:

"""
print(bankdata.shape)

""""Exploring the Dataset
We've just seen that there are only zeros and ones in the class column, but we can also know in what proportion they are - in other words - if there are more 
zeros than ones, more ones than zeros, or if the numbers of zeros is the same as the number of ones, meaning they are balanced"""

print ( " Exploring the Dataset:  bankdata['class'].value_counts()) \n " , bankdata['class'].value_counts())

""""
The first step is to use pandas value_counts() method again, but now let's see the percentage by including the argument normalize=True:

The normalize=True calculates the percentage of the data for each class. So far, the percentage of forged (0) and real data (1) is:

"""

print ( " Exploring the Dataset:  bankdata['class'].value_counts()) \n " , bankdata['class'].value_counts(normalize=True) )


"""
We can also see this difference visually, by taking a look at the class or target's distribution with a Pandas imbued histogram, by using:
"""

bankdata['class'].plot.hist();
plt.show()
#plt.close()

"""
We can have a look at the statistical measurements with the describe() dataframe method. We can also use .T of transpose - to invert columns and rows, making it more direct to compare across values:
"""

print("bankdata.describe().T   :    \n" , bankdata.describe().T )

"""
Let's start with each feature's distribution, and plot the histogram of each data column except for the class column. The class column will not be taken into consideration by its position in the bankdata columns array. All columns will be selected except for the last one with columns[:-1]:
"""

import matplotlib.pyplot as plt

for col in bankdata.columns[:-1]:
    plt.title(col)
    gc= bankdata[col].plot.hist() #plotting the histogram with Pandas
    gc.figure.show()
    #plt.show();


#plt.close()

"""
We can now move on to the second part, and plot the scatter plot of each variable. To do this, we can also select all columns except for the class, with columns[:-1],
 use Seaborn's scatterplot() and two for loops to obtain the variations in pairing for each of the features. We can also exclude the pairing of a feature with itself, 
 by testing if the first feature equals the second one with an if statement.

"""
import seaborn as sns
import matplotlib.pyplot as plt

"""
for feature_1 in bankdata.columns[:-1]:
    for feature_2 in bankdata.columns[:-1]:
        if feature_1 != feature_2: # test if the features are different
            print(feature_1, feature_2) # prints features names
            gi= sns.scatterplot(x=feature_1, y=feature_2, data=bankdata, hue='class') # plots each feature points with its color depending on the class column value
            gi.figure.show() 
"""

#plt.close()


"""
But looking at all of those graphs in sequence can be a little hard. We have the alternative of looking at all the distribution and scatter plot graphs together by using Seaborn's pairplot().

Both previous for loops we had done can be substituted by just this line:
"""
sns.pairplot(bankdata, hue='class');
plt.show()

#plt.close()


"""
To separate the target and features, we can attribute only the class column to y, later dropping it from the dataframe to attribute the remaining columns to X with .drop() method:

"""

y = bankdata['class']
X = bankdata.drop('class', axis=1) # axis=1 means dropping from the column axis


""""
To use it, we can import the library, call the train_test_split() method, pass in X and y data, and define a test_size to pass as an argument. In this case, we will define it as 0.20- this means 20% of the data will be used for testing, and the other 80% for training.

This method randomly takes samples respecting the percentage we've defined, but respects the X-y pairs, lest the sampling would totally mix up the relationship.

Since the sampling process is inherently random, we will always have different results when running the method. To be able to have the same results, or reproducible results, we can define a constant called SEED with the value of 42.

You can execute the following script to do so:

from sklearn.model_selection import train_test_split
"""
from sklearn.model_selection import train_test_split

SEED = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = SEED)


"""Notice that the train_test_split() method already returns the X_train, X_test, y_train, y_test sets in this order. We can print the number of samples separated for train and test by getting the first (0) element of the shape property returned tuple:"""

xtrain_samples = X_train.shape[0]
xtest_samples = X_test.shape[0]

print(f'There are {xtrain_samples} samples for training and {xtest_samples} samples for testing.')



""""Training the Model
We have divided the data into train and test sets. Now it is time to create and train an SVM model on the train data. To do that, we can import Scikit-Learn's svm library along with the Support Vector Classifier class, or SVC class.

After importing the class, we can create an instance of it - since we are creating a simple SVM model, we are trying to separate our data linearly, so we can draw a line to divide our data - which is the same as using a linear function - by defining kernel='linear' as an argument for the classifier:
"""
from sklearn.svm import SVC
svc = SVC(kernel='linear')

"""This way, the classifier will try to find a linear function that separates our data. After creating the model, let's train it, or fit it with the train data,
 employing the fit() method and giving the X_train features and y_train targets as arguments.

We can execute the following code in order to train the model:"""

svc.fit(X_train, y_train)

"""
Making Predictions
A way to answer if the model managed to describe the data is to calculate and look at some classification metrics.

Considering that the learning is supervised, we can make predictions with X_test and compare those prediction results - which we might call y_pred - with the actual y_test, or ground truth

"""
y_pred = svc.predict(X_test)



"""For a better visualization of the confusion matrix, we can plot it in a Seaborn's heatmap along with quantity annotations, and for the classification report, 
it is best to print its outcome, so its results are formatted. This is the following code:"""

from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(y_test,y_pred)
gg=sns.heatmap(cm, annot=True, fmt='d').set_title('Confusion matrix of linear SVM') # fmt='d' formats the numbers as digits, which means integers
gg.figure.show() 

#plt.close()

print(classification_report(y_test,y_pred))


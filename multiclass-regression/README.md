# Softmax Regression Implementation 

Writing a separate package compatible with scikit-learn, to compute softmax regression with regularization.

Usage:
```python
from classification import SoftMaxClassifier
# Some data
X = [[1,1,1], [0,0,0]]
y = [0, 1]
clf = SoftMaxClassifier()
clf.fit(X, y)
clf.predict((X[0])) # index pf class with max prob
clf.predict_prob((X[0])) # list of prob for each class
clf.coef_
```

Reference: 
[Softmax Regression](http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/), [Matrix equations](https://towardsdatascience.com/multiclass-logistic-regression-from-scratch-9cc0007da372#:~:text=Multiclass%20logistic%20regression%20is%20also,really%20know%20how%20it%20works.)
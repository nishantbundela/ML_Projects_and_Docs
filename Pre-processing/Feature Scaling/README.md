# Data Preprocessing Project – Feature Scaling

 **Feature Scaling** is the process used to standardize range of independent variables so that they can be mapped onto same scale. In this project, I discuss various data preprocessing techniques related to Feature Scaling. I have discussed useful estimators related to Feature Scaling. The estimators are MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, Normalizer, Binarizer and scale.


===============================================================================


## Table of Contents

The contents of this project are divided as follows:-

1.	Introduction


2.	Rescaling data with MinMaxScaler


3.	Standardising data with StandardScaler


4.	Rescaling data with MaxAbsScaler


5.	Rescaling data with RobustScaler


6.	Normalizing data with Normalizer


7.	Binarizing data with Binarizer


8.	Mean removal with scale


9.	Applications


10.	References

===============================================================================

## 1. Introduction


One of the most important data preprocessing step, we need to apply to our data is feature scaling. When we encounter any 
real world data set, the independent or feature variables may be mapped onto different scales.  Feature scaling refers to 
procedures used to standardize these independent or feature variables so that they are mapped onto same scales.


Most of the ML algorithms perform well when the feature variables are mapped onto the same scale. They don’t perform well 
when features are mapped onto different scales. For example, in stochastic gradient descent, feature scaling can improve 
the convergence speed of the algorithm. In support vector machines, it can reduce the time to find support vectors. 


But, there are few exceptions as well. Decision trees and random forests are two of the algorithms where we don’t need to 
worry about feature scaling. These algorithms are scale invariant. Similarly, Naive Bayes and Linear Discriminant Analysis 
are not affected by feature scaling. In Short, any Algorithm which is not distance based is not affected by feature scaling.


So, let’s start our discussion of various techniques associated with feature scaling.


===============================================================================

## 2. Rescaling data with MinMaxScaler


This technique of rescaling is also called **min-max scaling** or **min-max normalization**. **Normalization** refers to the 
rescaling of the features to a range of [0, 1], which is a special case of min-max scaling. So, in this technique, values are 
shifted and rescaled so that they end up ranging from zero to one. We do this by subtracting the minimum value (xmin ) and 
dividing by the maximum value (xmax ) minus the minimum value (xmin ).


Mathematically, the new value x(i)norm of a sample x(i) can be calculated as follows:-

 
		     x(i)norm  =  (xi-  xmin )/(xmax-  xmin )
		     
	
Here, x(i) is a particular sample value. xmax and xmin is the maximum and minimum feature value in a column.

	
Scikit-Learn provides a transformer called **MinMaxScaler** for this task. It has a feature range parameter to adjust the range of values. This estimator fit and transform each feature variable individually such that it is in the given range (between zero and one)
on the training set. 

As with all the other transformers, we fit this transformer to the training data only, not to the full data set (including the test set). Only then we can use them to transform the training set and the test set and new data.


The syntax for implementing **min-max scaling** procedure in Scikit-Learn is given as follows:- 


`from sklearn.preprocessing import MinMaxScaler`

`ms = MinMaxScaler()`

`X_train_ms = ms.fit_transform(X_train)`

`X_test_ms = ms.transform(X_test)`


===============================================================================

## 3. Standardising data with StandardScaler


There is another practical approach for feature scaling which might be more useful in certain circumstances. It is called **standardization**. It can be more useful for many machine learning algorithms, especially for optimization algorithms such 
as gradient descent.


In **standardization**, first we determine the distribution mean and standard deviation for each feature. Next we subtract the 
mean from each feature. Then we divide the values of each feature by its standard deviation. So, in standardization, we center 
the feature columns at mean 0 with standard deviation 1 so that the feature columns takes the form of a normal distribution, 
which makes it easier to learn the weights. 


Mathematically, **standardization** can be expressed by the following equation: 


		x(i)std =  ( x(i)- μx)/(σx )


Here, x(i) is a particular sample value and x(i)std is its standard deviation , μx is the sample mean of a particular 
feature column and σx is the corresponding standard deviation.


**Min-max scaling** scales the data to a limited range of values. Unlike min-max scaling, **standardization** does not bound 
values to a specific range. So, standardization is much less affected by outliers. Standardization maintains useful information 
about outliers and is much less affected by them. It makes the algorithm less sensitive to outliers in contrast to min-max scaling. 


Scikit-Learn provides a transformer called **StandardScaler** for standardization. The syntax to implement standardization 
is quite similar to min-max scaling given as follows :-


`from sklearn.preprocessing import StandardScaler`

`ss = StandardScaler()`

`X_train_ss = ss.fit_transform(X_train)`

`X_test_ss = ss.transform(X_test)`


Again, we should fit the StandardScaler class only once on the training data set and use those parameters to transform the 
test set or new data set.

===============================================================================

## 4. Rescaling data with MaxAbsScaler


In this feature rescaling task, we rescale each feature by its maximum absolute value. So, the maximum absolute value of each 
feature in the training set will be 1.0. It does not affect the data and hence there is no effect on sparsity.


Scikit-Learn provides **MaxAbsScaler** transformer for this task.


The syntax for implementing max-abs scaling procedure in Scikit-Learn is given as follows:- 


`from sklearn.preprocessing import MaxAbsScaler`

`mabs = MaxAbsScaler()`

`X_train_mabs = mabs.fit_transform(X_train)`

`X_test_mabs = mabs.transform(X_test)`

===============================================================================

## 5. Rescaling data with RobustScaler


**StandardScaler** can often give misleading results when the data contain outliers.  Outliers can often influence the 
sample mean and variance and hence give misleading results. In such cases, it is better to use a scalar that is robust 
against outliers. 


Scikit-Learn provides a transformer called **RobustScaler** for this purpose.


The **RobustScaler** is very similar to **MinMaxScaler**. The difference lies in the parameters used for scaling. 
While **MinMaxScaler** uses minimum and maximum values for rescaling, **RobustScaler** uses interquartile(IQR) 
range for the same.



Mathematically, the new value x(i)norm of a sample x(i) can be calculated as follows:-
 
		     x(i)  =  (xi-  Q1(x) )/(Q3(x) - Q1(x))
		     
		     

Here, x(i) is the scaled value, xi is a particular sample value, Q1(x) and Q3(x) are the 1st quartile (25th quantile) and 
3rd quartile (75th quantile) respectively. So, Q3(x) - Q1(x) is the difference between 3rd quartile (75th quantile) and 
1st quartile (25th quantile) respectively. It is called IQR (Interquartile Range).

	

The syntax for implementing scaling using RobustScaler in Scikit-Learn is given as follows:- 


`from sklearn.preprocessing import RobustScaler`

`rb = RobustScaler()`

`X_train_rb = rb.fit_transform(X_train)`

`X_test_rb = rb.transform(X_test)`

===============================================================================

## 6. Normalizing data with Normalizer

In this feature scaling task, we rescale each observation to a length of 1 (a unit norm). Scikit-Learn provides the 
**Normalizer** class for this task. In this task, we scale the components of a feature vector such that the complete 
vector has length one. This usually means dividing each component by the Euclidean length (magnitude) of the vector.



Mathematically, normalization can be expressed by the following equation: 



		x(i)norm =   x(i) / | x(i)|
		

where  x(i) is a particular sample value , x(i)norm is its normalized value and | x(i)| is the corresponding 
Euclidean length of the vector. 



The syntax for normalization is quite similar to standardization given as follows:-


`from sklearn.preprocessing import Normalizer`

`nm = Normalizer()`

`X_train_nm = nm.fit_transform(X_train)`

`X_test_nm = nm.transform(X_test)`

===============================================================================

## 7. Binarizing data with Binarizer

In this feature scaling procedure, we binarize the data (set feature values equal to 0 or 1) according to a threshold. 
So, using a binary threshold, we transform our data by marking the values above it to 1 and those equal to or below it 
to 0. Scikit-Learn provides **Binarizer** class for this purpose. 


The syntax for binarizing the data follow the same rules as above and is given below:-


`from sklearn.preprocessing import Binarizer`

`binr = Binarizer()`

`X_train_binr = binr.fit_transform(X_train)`

`X_test_binr = binr.transform(X_test)`

===============================================================================

## 8. Mean removal with scale

In this feature scaling task, we remove the mean from each feature to centre it on zero. Thus, we standardize a dataset 
along any axis. 


Scikit-Learn provides **scale** class for this purpose. 


The syntax for this purpose is given below:-


`from sklearn.preprocessing import scale`

`scl = scale()`

`X_train_scl = scl.fit_transform(X_train)`

`X_test_scl = scl.transform(X_test)`

===============================================================================

## 9. Applications

Now, I will discuss few applications of feature scaling.

Generally real world dataset contains features that differ in magnitudes, units and range. We should perform **Normalization**
when the scale of a feature is irrelevant or misleading. The algorithms which depend on Euclidean distance measure are sensitive 
to magnitudes. In this case, feature scaling helps to weigh all the features equally.


Suppose a feature in the dataset is relatively big in magnitude as compared to other features. Then in algorithms where Euclidean 
distance is measured, this big scaled feature becomes dominating and needs to be normalized.


### Applications of Feature Scaling 


I will discuss algorithms where feature scaling matters. These are as follows:-
 

1. **K-Means** – K Means algorithm is based on the Euclidean distance measure.  Hence feature scaling matters.


2. **K-Nearest-Neighbours** also require feature scaling. 


3. **Principal Component Analysis (PCA)**: In PCA algorithm, we try to get the feature with maximum variance. 
Here too feature scaling is required.


4. **Gradient Descent**: In gradient descent algorithm, calculation speed increases as theta calculation becomes faster 
after feature scaling.


I have previously stated that any algorithm which is not distance based is not affected by feature scaling. 


So, **Naive Bayes**, **Linear Discriminant Analysis** and **Tree-Based models (Decision Trees and Random Forests)** 
are not affected by feature scaling.


===============================================================================

## 10. References

The ideas and techniques in this project have been taken from the following books and websites:-

1. Scikit-Learn API reference

2. Python Machine Learning by Sebastian Raschka

3. Hands-On Machine Learning with Scikit Learn and Tensorflow by Aurélien Géron 

4. https://en.wikipedia.org/wiki/Feature_scaling

5. https://sebastianraschka.com/Articles/2014_about_feature_scaling.html






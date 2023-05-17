# k-Nearest Neighbours with Python and Scikit-Learn


k Nearest Neighbours is a very simple and one of the topmost machine learning algorithms. In this project, I build a k Nearest Neighbours classifier to classify the patients suffering from Breast Cancer. I have used the `Breast Cancer Wisconsin (Original) Data Set` downloaded from the UCI Machine Learning Repository.


===============================================================================


## Table of Contents


I have categorized this project into various sections which are listed below:-


1.	Introduction to k-Nearest Neighbours (kNN) algorithm
2.	k-Nearest Neighbours (kNN) intuition
3.	kNN algorithm in action
4.	How to decide the number of neighbours in kNN
5.	Eager learners vs lazy learners
6.	Advantages and disadvantages of kNN algorithm
7.	The problem statement
8.	Results and conclusion
9.	Applications of kNN
10.	References


===============================================================================


## 1. Introduction to k Nearest Neighbours (kNN) algorithm

In machine learning, kNN is the simplest of all machine learning algorithms. It is a non-parametric algorithm used for classification and regression tasks. Non-parametric means there is no assumption required for data distribution. So, kNN does not require any underlying assumption to be made. In both classification and regression tasks, the input consists of the k closest training examples in the feature space. The output depends upon whether kNN is used for classification or regression purposes.

-	In kNN classification, the output is a class membership. The given data point is classified based on the majority of type of its neighbours. The data point is assigned to the most frequent class among its k nearest neighbours. Usually k is a small positive integer. If k=1, then the data point is simply assigned to the class of that single nearest neighbour.

-	In kNN regression, the output is simply some property value for the object. This value is the average of the values of k nearest neighbours.


kNN is a type of instance-based learning or lazy learning. Lazy learning means it does not require any training data points for model generation. All training data will be used in the testing phase. This makes training faster and testing slower and costlier. So, the testing phase requires more time and memory resources.

In kNN, the neighbours are taken from a set of objects for which the class or the object property value is known. This can be thought of as the training set for the kNN algorithm, though no explicit training step is required. In both classification and regression kNN algorithm, we can assign weight to the contributions of the neighbours. So, nearest neighbours contribute more to the average than the more distant ones.


===============================================================================


## 2. k Nearest Neighbours (kNN) intuition

The kNN algorithm intuition is very simple to understand. It simply calculates the distance between a sample data point and all the other training data points. The distance can be Euclidean distance or Manhattan distance. Then, it selects the k nearest data points where k can be any integer. Finally, it assigns the sample data point to the class to which the majority of the k data points belong.


### Euclidean distance

In mathematics, the Euclidean distance or Euclidean metric is the ordinary straight-line distance between two points in Euclidean space. The distance between the two points will then be calculated as the length of the hypotenuse of the two points. 

Euclidean Distance between any pair of points (x1, y1) and (x2, y2) is given by


`[(x2-x1)^2 + (y2-y1)^2]^0.5`


### Manhattan distance

Manhattan distance between any two points is the sum of the absolute differences of their cartesian coordinates. 
Manhattan Distance between two points (x1, y1) and (x2, y2) is given by –


`|x1 – x2| + |y1 – y2|`


===============================================================================


## 3. kNN algorithm in action


Now, I will see kNN algorithm in action. Suppose, we have a dataset with two variables, which when plotted, looks like the following figure.


![Scatter-plot of 2 variables](https://github.com/pb111/K-Nearest-Neighbours-Project/blob/master/Images/kNN%201.png)


In kNN algorithm, k is the number of nearest neighbours. Generally, k is an odd number because it helps to decide the majority of the class. When k=1, then the algorithm is known as the nearest neighbour algorithm.

Now, we want to classify a new data point `X` into `Blue` class or `Red` class. Suppose the value of k is 3. The kNN algorithm starts by calculating the distance between `X` and all the other data points. It then finds the 3 nearest points with least distance to point `X`. This is clearly shown in the figure below. The 3 nearest points have been encircled.


![kNN algorithm in action](https://github.com/pb111/K-Nearest-Neighbours-Project/blob/master/Images/kNN%202.png)

In the final step of the kNN algorithm, we assign the new data point `X` to the majority of the class of the 3 nearest points. From the above figure, we can see that the 2 of the 3 nearest points belong to the class `Red` while 1 belong to the class `Blue`. Therefore, the new data point will be classified as `Red`.


===============================================================================

## 4. How to decide the number of neighbours in kNN


While building the kNN classifier model, one question that come to my mind is what should be the value of nearest neighbours (k) that yields highest accuracy. This is a very important question because the classification accuracy depends upon our choice of k.


The number of neighbours (k) in kNN is a parameter that we need to select at the time of model building. Selecting the optimal value of k in kNN is the most critical problem. A small value of k means that noise will have higher influence on the result. So, probability of overfitting is very high. A large value of k makes it computationally expensive in terms of time to build the kNN model. Also, a large value of k will have a smoother decision boundary which means lower variance but higher bias.


The data scientists choose an odd value of k if the number of classes is even. We can apply the elbow method to select the value of k. To optimize the results, we can use Cross Validation technique. Using the cross-validation technique, we can test the kNN algorithm with different values of k. The model which gives good accuracy can be considered to be an optimal choice. It depends on individual cases and at times best process is to run through each possible value of k and test our result. 


===============================================================================


## 5. Eager learners vs lazy learners


`Eager learners` mean when giving training data points, we will construct a generalized model before performing prediction on given new points to classify. We can think of such learners as being ready, active and eager to classify new data points. 


`Lazy learning` means there is no need for learning or training of the model and all of the data points are used at the time of prediction. Lazy learners wait until the last minute before classifying any data point. They merely store the training dataset and waits until classification needs to perform. Lazy learners are also known as `instance-based learners` because lazy learners store the training points or instances, and all learning is based on instances.


Unlike eager learners, lazy learners do less work in the training phase and more work in the testing phase to make a classification. 


===============================================================================


## 6. Advantages and disadvantages of kNN algorithm


The advantages of kNN algorithm are as follows:-

1.	kNN is simple to implement.
2.	It executes quickly for small training datasets.
3.	It does not need any prior knowledge about the structure of the data in training set.
4.	kNN is a lazy learning algorithm and therefore requires no training prior to making real time predictions. This makes the kNN algorithm much faster than other algorithms that require training, e.g. SVM, linear regression, etc.
5.	In kNN algorithm, no retraining is required if the new training data is added to the existing training set.
6.	There are only two parameters required to implement kNN. The value of k and the distance function (e.g. Euclidean or Manhattan distance etc.)


The disadvantages of kNN algorithm are as follows:-

1.	When the training set is large, it may take a lot of space and memory.
2.	For every test data, the distance should be computed between test data and all the training data samples. Thus, a lot of time may be needed for testing.
3.	Finally, the kNN algorithm doesn’t work well with categorical features since it is difficult to find distance between dimensions with categorical features.


===============================================================================


## 7. The problem statement


In this project, I try to classify the patients suffering from breast cancer. I implement kNN algorithm with Python and Scikit-Learn. 


To answer the question, I build a kNN classifier to predict whether or not a patient is suffering from breast cancer. I have used the **Breast Cancer Wisconsin (Original) Data Set** downloaded from the UCI Machine Learning Repository for this project.


===============================================================================


## 8. Results and conclusion


1. In this project, I build a kNN classifier model to classify the patients suffering from breast cancer. The model yields very good performance as indicated by the model accuracy which was found to be 0.9786 with k=7.

2. With k=3, the training-set accuracy score is 0.9821 while the test-set accuracy to be 0.9714. These two values are quite comparable. So, there is no question of overfitting. 

3. I have compared the model accuracy score which is 0.9714 with null accuracy score which is 0.6071. So, we can conclude that our K Nearest Neighbours model is doing a very good job in predicting the class labels.

4. Our original model accuracy score with k=3 is 0.9714. Now, we can see that we get same accuracy score of 0.9714 with k=5. But, if we increase the value of k further, this would result in enhanced accuracy. With k=6,7,8 we get accuracy score of 0.9786. So, it results in performance improvement. If we increase k to 9, then accuracy decreases again to 0.9714. So, we can conclude that our optimal value of k is 7.

5. kNN Classification model with k=7 shows more accurate predictions and less number of errors than k=3 model. Hence, we got performance improvement with k=7.

6. ROC AUC of our model approaches towards 1. So, we can conclude that our classifier does a good job in predicting whether it is benign or malignant cancer.

7. Using the mean cross-validation, we can conclude that we expect the model to be around 96.46 % accurate on average.

8. If we look at all the 10 scores produced by the 10-fold cross-validation, we can also conclude that there is a relatively high variance in the accuracy between folds, ranging from 100% accuracy to 87.72% accuracy. So, we can conclude that the model is very dependent on the particular folds used for training, but it also be the consequence of the small size of the dataset.


===============================================================================


## 9. Applications of kNN


kNN algorithm is used for both classification and regression problems. It is used in the variety of applications such as 


1.	Finance
2.	Healthcare
3.	Political science
4.	Handwriting detection
5.	Image recognition
6.	Video recognition
7.	Predicting the credit-rating of customers
8.	Predicting the probability of a loan repayment


===============================================================================


## 10. References


The work done in this project is inspired from following books and websites:-

1.	Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron

2.	Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido

3.	Udemy course – Machine Learning – A Z by Kirill Eremenko and Hadelin de Ponteves

4.	https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

5.	https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn

6.	http://dataaspirant.com/2016/12/23/k-nearest-neighbor-classifier-intro/

7.	https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/





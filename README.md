AI : Artificial intelligence (AI) is a set of technologies that allow computers to perform tasks that are usually associated 
with human intelligence.

ML : A branch of AI that uses mathematical models and statistical algorithms to teach computers to learn and improve without 
direct instruction.

ML Algorithms types
	1. Supervised Learning: Supervised learning is a category of machine learning that uses labeled datasets to train algorithms to predict outcomes and recognize patterns
	2. Unsupervised Learning
	3. Reinforcement Learning(RL)

Supervised Learning is a category of machine learning that uses labeled datasets to train algorithms to 
predict outcomes and recognize patterns.

Unsupervised learning is a type of machine learning that learns from data without human supervision. Unsupervised machine 
learning models are given unlabeled data and allowed to discover patterns and insights without any explicit guidance or 
instruction.

1. Supervised Learning
	Response
	|_Categorical(Classification)
	|	|__Single Label
	|	|	|__Binary     --> Yes/No, Male/Female
	|	|	|__Multiclass --> Red/Green/Blue/Black
	|	|
	|	|__Multi Label    --> Diabetes, Blood Pressure, Sugar
	|
	|_Numerical(Regression)
	
	Model is the process that uses some metrics which help us to analyze the performance of the model
	|__Categorical(Classification)
	|	|__Misclassification error
	|	|__Accuracy score (It cannot perform well on an imbalanced dataset)
	|	|__Recall
	|	|__Precision
	|	|__F1 score
	|	|__AUC(Area under Curve)
	|	|__ROC(Receiver opterating Characteristics)
	|	|__Log-loss
	|
	|__Numerical(Regression)
		|__Mean Absolute Error(MAE)
		|__Mean Squared Error(MSE)
		|__R2 Score

Accuracy is defined as the ratio of the number of correct predictions to the total number of predictions.It cannot perform 
well on an imbalanced dataset
The formula is given by Accuracy = (TP+TN)/(TP+TN+FP+FN)

Precision is the ratio of true positives to the summation of true positives and false positives. It basically analyses the positive predictions.
Precision = TP/(TP+FP)

Recall is the ratio of true positives to the summation of true positives and false negatives. It basically analyses the number of correct positive samples.
Recall = TP/(TP+FN)

The F1 score is the harmonic mean of precision and recall
F1 score = (2×Precision×Recall)/(Precision+Recall)

AUC (Area Under Curve) is an evaluation metric that is used to analyze the classification model at different threshold values. The Receiver 
Operating Characteristic(ROC) curve is a probabilistic curve used to highlight the model’s performance. This curve is useful as 
it helps us to determine the model’s capacity to distinguish between different classes. It basically highlights a model’s 
capacity to separate the classes. 0.75 is a good AUC score. A model is considered good if the AUC score is greater than 0.5 
and approaches 1. A poor model has an AUC score of 0.

	
	Data Pre-Processing
	|__Scaling
		|__Standard Scaler
		|__Min Max Scaler(0,1)
		
	Baye's Theorem 
	(The sklearn.naive_bayes module implements Naive Bayes algorithms. These are supervised learning methods based on applying 
	Bayes’ theorem with strong (naive) feature independence assumptions. Document classification and spam filtering, although 
	naive Bayes is known as a decent classifier, it is known to be a bad estimator, so the probability outputs from 
	predict_proba are not to be taken too seriously.)
	|
	|__Classification(Response = Categorical)
		|__Discrete NB/MultinomialNB -> Features = Categorical,  Response = Categorical
		|__Kernel NB/GaussianNB	  	 -> Features = Numerical,	 Response = Categorical
		|__BernoulliNB				 -> Features = Binary,		 Response = Binary
		
	K-Nearest Neighbors
	The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the 
	new point, and predict the label from these. The number of samples can be a user-defined constant (k-nearest neighbor 
	learning), or vary based on the local density of points (radius-based neighbor learning). The distance can, in general, 
	be any metric measure: standard Euclidean distance is the most common choice.
	|
	|__Classification(Response = Categorical)
	|	|_KNeighborsClassifier (neighbors.KNeighborsClassifier)
	|
	|_Regression(Response = Numerical)
		|_KNeighborsRegressor (neighbors.KNeighborsRegressor)
	
	Linear Model(fitting a line to data)
	|
	|__Regression(Response = Numerical)
	|	|_Linear Regression(Least sum of squares)
	|	|	|_Lasso
	|	|	|_Ridge
	|	|	|_ElasticNet
	|	|_SGDRegressor
	|	
	|__Classification(Response = Categorical)
		|_LogisticRegression --> Binary --> Sigmoid (maximum likelihood)
		|					 --> Multiclass --> Softmax
		|_RidgeClassifier
		|_SGDClassifier
		
	Discriminant Analysis(n-dimensional normal distribution, Featurs may depends on each others)
	Dimensionality Reduction technique, maximize separation of means(categories), minimum separation of variance
	|
	|__Classification(Response = Categorical)
		|_LinearDiscriminantAnalysis
		|_QuadraticDiscriminantAnalysis
		
	Support Vector Machines(maximum margin classifier, the shortest distance between the observation and threshold is margin.
	when we allow misclassification, the distance between the observation and threshold is called soft margin, determine threshold
	using soft margin is known as soft margin classifier/Support Vector classifier. The observations on the edge and within 
	soft margin are called Support Vector)
	|
	|__Regression(Response = Numerical)
	|	|_SVR(Epsilon-Support Vector Regression)
	|
	|__Classification(Response = Categorical)
		|_SVC(C-Support Vector Classification)
		
	Decision Trees
	|
	|__Regression(Response = Numerical)
	|	|_DecisionTreeRegressor
	|
	|__Classification(Response = Categorical)
		|_DecisionTreeClassifier --> Gini's Impurity Index
	
	Ensemble Learning
	|
	|__Voting
	|	|
	|	|__Regression(Response = Numerical)
	|	|	|__Averaging
	|	|	|__Weighted Averaging
	|	|
	|	|__Classification(Response = Categorical)
	|		|__Max Voting
	|
	|__Bagging(Bootstrap Aggregating)
	|	|
	|	|__Regression(Response = Numerical)
	|	|	|__BaggingRegressor
	|	|
	|	|__Classification(Response = Categorical)
	|		|__BaggingClassifier
	|			|__Random forest
	|
	|__Boosting
	|	|__AdaBoost(_Adaptive Boost) (DecisionTree with max_depth = 1) <-- weak learner
	|	|	|__Regression(Response = Numerical)
	|	|	|	|__AdaBoostRegressor
	|	|	|
	|	|	|__Classification(Response = Categorical)
	|	|		|_AdaBoostClassifier
	|	|
	|	|__Gradient Boosting Method
	|	|	|__Regression(Response = Numerical)
	|	|	|	|__GradientBoostingRegressor
	|	|	|
	|	|	|__Classification(Response = Categorical)
	|	|		|__GradientBoostingClassifier
	|	|
	|	|__XGradient Boosting Method
	|		|__Regression(Response = Numerical)
	|		|	|__XGBRegressor
	|		|
	|		|__Classification(Response = Categorical)
	|			|__XGBClassifier
	|__Stacking
		|__Regression(Response = Numerical)
			|	|__StackingRegressor
			|
			|__Classification(Response = Categorical)
				|____Stackinglassifier
		
		
	Time Series Analysis(Regression)
	|
	|__Smoothing Method
	|	|__Moving Average
	|	|	|__Centered Moving Average(Visualization)
	|	|	|__Trailing Moving Average(Forecasting)
	|	|
	|	|__SimpleExponentialSmoothing(smoothing_level=alpha)
	|	|
	|	|__Holt's Method (Holt)
	|	|	|__LINEAR Trend -- (smoothing_level=alpha, smoothing_trend=beta)
	|	|	|__Exponential Trend -- (exponential=True, smoothing_level=alpha, smoothing_trend=beta)
	|	|	|__Additive Damped Trend -- (damped_trend=True, smoothing_level=alpha, smoothing_slope=phi)
	|	|	|__Multiplicative Damped Trend -- (exponential=True, damped_trend=True, smoothing_level=alpha, smoothing_slope=phi)
	|	|
	|	|__Holt-Winter's Method(ExponentialSmoothing)
	|		|__Additive Trend -- (seasonal_periods=len(y_test), trend='add', seasonal='add')
	|		|__Additive & Damped Trend -- (damped_trend=True, seasonal_periods=len(y_test), trend='add', seasonal='add')
	|		|__Multiplicative Trend  -- (seasonal_periods=len(y_test), trend='add', seasonal='mul')
	|		|__Additive & DampedTrend -- (damped_trend=True, seasonal_periods=len(y_test), trend='add', seasonal='mul')
	|
	|__ARIMA(Auto-Regressive Integrated with Moving Average)(p,d,q)
		Auto-Regressive Model
		Moving Average Model
		ARMA(Auto-Regressive Moving Average)(p,q)
		Seasonal ARIMA(p,d,q)(P,D,Q)[S]
	
2. Unsupervised Learning
	|
	|__Cluster Analysis(Clustering/Grouping of Numerical range or category or any of the feature)
	|	|__Hierarchical --> Visualization(for small dataset)
	|	|	|__Agglomerative
	|	|		|__Linkage(Single: Smallest Euclidean distance between 2 points of 2 different clusters, 
	|	|				   Complete: Largest Euclidean distance between 2 points of 2 different clusters,
	|	|				   Centroid: Euclidean distance between 2 mid points of 2 different clusters,
	|	|				   Average: Average of all Euclidean distances from each point of one cluster to another cluster)
	|	|
	|	|__Non-hierarchical
	|		|__K-Means
	|		|	|__KMeans(K randomly placed points start forming clusters until Convergence happens)
	|		|__DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 
	|			|__DBSCAN (sklearn.cluster import DBSCAN) 
	|				eps = The maximum distance between two samples for one to be considered as in the neighborhood of the other.
	|				min_sample = The number of samples (or total weight) in a neighborhood for a point to be considered as a 
	|				core point. This includes the point itself.
	|
	|__Dimensionality Reduction
	|	|__Singular Value Decomposition
	|	|__Principle Component Analysis
	|
	|__Assotion Rules Mining(mlxtend.frequent_patterns import association_rules,apriori)
	|		Support : Number of transactions that include the item or items
	|		Permissible Support : Allowed Support to work with the data
	|		Confidence : (Support(if purchased) and Support(than purchase))/Support(than purchase)
	|		Benchmark Confidence : Support(than purchase)/ Number of total transactions
	|		Lift Ratio : Confidence/Benchmark Confidence, should be >1 to work with
	|
	|__Recommender System
		Personalized(Ads)
		Non Personalized
			Content-Based Filtering
			User-Based Collaborative Filtering (User-Item Matrix)
			Item-Based Collaborative Filtering (Item-Item Matrix)
			
3. NEURAL NETWORKS can be used for both supervised and unsupervised learning, as well as reinforced learning.
tf.random.set_seed(2024)
model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(4, activation='relu',input_shape=(X_train.shape[1],)), 
		tf.keras.layers.Dense(1, activation='sigmoid') ])
1)	ACTIVATION functions:
	https://medium.com/@ssiddharth408/the-most-commonly-used-activation-functions-in-tensorflow-v2-11-2132107a440
	The activation function decides whether a neuron should be activated or not by calculating the weighted sum and further 
	adding bias to it. The purpose of the activation function is to introduce non-linearity into the output of a neuron.
	In a neural network, we would update the weights and biases of the neurons on the basis of the error at the output.
	This process is known as back-propagation. Activation functions make the back-propagation possible.
	
	Why do we need a Non-linear activation function?
	A neural network without an activation function is essentially just a linear regression model. The activation function 
	does the non-linear transformation to the input making it capable to learn and perform more complex tasks.
	
	A. ReLU (Rectified Linear Unit): A simple activation function that sets all negative values to zero and leaves all positive 
	values as they are, max(x, 0). This function is commonly used in deep neural networks as it can help prevent the vanishing gradient 
	problem.
	
	B. Leaky ReLU: A variation of ReLU that allows a slight gradient when the input is negative. This helps to avoid “dead” 
	neurons that have a zero output and never activate again.
	
	C. ELU (Exponential Linear Unit): Another variation of ReLU that uses the exponential function for negative inputs, resulting 
	in a smoother transition from negative to positive values (x, -1).
	
	D. SELU (Scaled Exponential Linear Unit): A self-normalizing version of the ELU activation function that helps to prevent 
	exploding or vanishing gradients in deep neural networks.
	
	E. Softmax: Softmax is commonly used in the output layer of a neural network for multi-class classification problems.
	
	F. Sigmoid: Sigmoid maps any input value to a value between 0 and 1. Used in the output layer for binary classification problems. 
	
	G. Tanh (Hyperbolic Tangent): Similar to sigmoid, but maps input values to a range between -1 and 1. Tanh is also used in the 
	output layer for binary classification problems.
	
	
2)	OPTIMIZERS are techniques or algorithms used to decrease loss (an error) by tuning various parameters and weights, hence 
	minimizing the loss function, providing better accuracy of model faster.
	https://www.geeksforgeeks.org/optimizers-in-tensorflow/
	
	A. SGD: The stochastic Gradient Descent (SGD) optimization method executes a parameter update for every training example. 
	In the case of huge datasets, SGD performs redundant calculations resulting in frequent updates having high variance 
	causing the objective function to vary heavily.
	Advantages: 
		Requires Less Memory.
		Frequent alteration of model parameters.
		If Momentum is used then helps to reduce noise.
	Disadvantages: 
		High Variance
		Computationally Expensive
	
	B. AdaGrad:  Adaptive Gradient Algorithm. AdaGrad optimizer modifies the learning rate particularly with individual 
	features .i.e. some weights in the dataset may have separate learning rates than others.
	Advantages:
		Best suited for Sparse Dataset
		Learning Rate updates with iterations
	Disadvantages:
		Learning rate becomes small with an increase in depth of neural network
		May result in dead neuron problem

	C. RMSprop: Root Mean Square Propagation. RMSprop optimizer doesn’t let gradients accumulate for momentum instead only 
	accumulates gradients in a particular fixed window. It can be considered as an updated version of AdaGrad with few 
	improvements. RMSprop uses simple momentum instead of Nesterov momentum.
	Advantages:
		The learning rate is automatically adjusted.
		The discrete Learning rate for every parameter
	Disadvantage: 
		Slow learning
		
	D. Adadelta: Adaptive Delta optimizer is an extension of AdaGrad (similar to RMSprop optimizer), however, Adadelta 
	discarded the use of learning rate by replacing it with an exponential moving mean of squared delta (difference between 
	current and updated weights). It also tries to eliminate the decaying learning rate problem.
	Advantage:
		Setting of default learning rate is not required.
	Disadvantage: 
		Computationally expensive
		
	E. Adam: Adaptive Moment Estimation (Adam) is among the top-most optimization techniques used today. In this method, the 
	adaptive learning rate for each parameter is calculated. This method combines advantages of both RMSprop and momentum .
	i.e. stores decaying average of previous gradients and previously squared gradients.
	Advantages:
		Easy Implementation
		Requires less memory
		Computationally efficient
	Disadvantages:
		Can have weight decay problem
		Sometimes may not converge to an optimal solution
		
	F. AdaMax: AdaMax is an alteration of the Adam optimizer. It is built on the adaptive approximation of low-order moments 
	(based off on infinity norm). Sometimes in the case of embeddings, AdaMax is considered better than Adam.
	Advantages: 
		Infinite order makes the algorithm stable.
		Requires less tuning on hyperparameters
	Disadvantage: 
		Generalization Issue

	G. NAdam: NAdam is a short form for Nesterov and Adam optimizer. NAdam uses Nesterov momentum to update gradient than 
	vanilla momentum used by Adam. 
	Advantages: 
		Gives better results for gradients with high curvature or noisy gradients.
		Learns faster
	Disadvantage: 
		Sometimes may not converge to an optimal solution
		
3)	LOSS: We also called it an error function or cost function. These are the errors made by machines at the time of training 
	the data and using an optimizer and adjusting weight machines can reduce loss and can predict accurate results.
	https://www.analyticsvidhya.com/blog/2021/05/guide-for-loss-function-in-tensorflow/
	
	Probabilistic Loss Functions:
		1. Binary Cross-Entropy Loss
			Binary cross-entropy is used to compute the cross-entropy between the true labels and predicted outputs. 
			It’s used when two-class problems arise like cat and dog classification [1 or 0].
		
		2. Categorical Crossentropy Loss:
			The Categorical crossentropy loss function is used to compute loss between true labels and predicted labels.
			It’s mainly used for multiclass classification problems. For example Image classification of animal-like 
			cat, dog, elephant, horse, and human.
			
		3. Sparse Categorical Crossentropy Loss:
			It is used when there are two or more classes present in our classification task. similarly to categorical 
			crossentropy. But there is one minor difference, between categorical crossentropy and sparse categorical 
			crossentropy that’s in sparse categorical cross-entropy labels are expected to be provided in integers.
			Rather than using Sparse Categorical crossentropy we can use one-hot-encoding and convert the above problem into 
			categorical crossentropy.

		4.  Poisson loss:
			The poison loss is the mean of elements of tensor. we can calculate poison loss like y_pred – y_true*log(y_true)

		5. Kullback-Leibler Divergence Loss:
			Also, called KL divergence, it’s calculated by doing a negative sum of probability of each event P and then 
			multiplying it by the log of the probability of an event.
			KL(P || Q) = – sum x in X P(x) * log(Q(x) / P(x))

	Regression Losses: 
		6. Means Squared Error (MSE):
			MSE tells, how close a regression line from predicted points. And this is done simply by taking distance from 
			point to the regression line and squaring them. The squaring is a must so it’ll remove the negative sign problem.

		7. Mean Absolute Error:
			MAE simply calculated by taking distance from point to the regression line. The MAE is more sensitive to outliers. 
			So before using MAE confirm that data doesn’t contain outliers.

		8. Cosine Similarity Loss:
			Cosine similarity is a measure of similarity between two non-zero vectors. This loss function calculates the 
			cosine similarity between labels and predictions.
			It’s just a number between 1 and -1
			When it’s a negative number between -1 and 0 then, 0 indicates orthogonality, and values closer to -1 show greater 
			similarity.
	
		9. Huber Loss:
			The Huber loss function is quadratic for small values and linear for larger values,
			For each value of X the error = y_true-y_predLoss = 0.5 * X^2
			if |X| <= d Loss = 0.5 * d^2 + d (|X| – d)
			if |X| > d
		
		10. LogCosh Loss:
			The LogCosh loss computes the log of the hyperbolic cosine of the prediction error.

		Hinge Losses for ‘Maximum – Margin’ Classification:
		11. Hinge Loss
			It’s mainly used for problems like maximum-margin most notably for support vector machines.
			In Hinge loss values are expected to be -1 or 1. In the case of binary i.e. 0 or 1 it’ll get converted 
			into -1 and 1.

		12. Squared Hinge Loss:
			The Square Hinge loss is just square of hinge loss.

		13. Categorical Hinge Loss:
			It calculates the categorical hing loss between y_true and y_pred labels.


	model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])
	
	model.fit( X_train,y_train,validation_data=(X_test,y_test),verbose=2,epochs=500)
		Epoch 1/500
		16/16 - 4s - loss: 0.6942 - accuracy: 0.5992 - val_loss: 0.6884 - val_accuracy: 0.6476 - 4s/epoch - 251ms/step
			train_set_loss		train_set_accuracy		test_set_loss 		test_set_accuracy
	
	from tensorflow.keras.callbacks import EarlyStopping
	monitor = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=2, mode='auto',restore_best_weights=True)
	model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks=[monitor],verbose=2,epochs=500)

#### L1 L2 Regularizer ####	
tf.random.set_seed(seed=2024)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(25, activation='relu',input_shape=(X_train.shape[1], ),activity_regularizer = keras.regularizers.l1_l2(0.01))), 
    tf.keras.layers.Dense(20, activation='relu',activity_regularizer = keras.regularizers.l1_l2(0.01))
    tf.keras.layers.Dense(10, activation='relu',activity_regularizer = keras.regularizers.l1_l2(0.01))
    tf.keras.layers.Dense(5, activation='relu',activity_regularizer = keras.regularizers.l1_l2(0.01))
    tf.keras.layers.Dense(1,activation='sigmoid')])

####Drop Out Regularization####
Dropout is a regularization technique in deep learning that helps prevent overfitting in neural networks: 
How it works?
During training, dropout randomly deactivates a percentage of neurons in a neural network, usually in the hidden layers. 
Why it works?
Dropout prevents overfitting by making the network less reliant on specific neurons or features. It also ensures that no units are codependent on each other. 
When it's used?
Dropout is only active during the training phase. 
How it's similar to sexual reproduction?
Dropout is similar to sexual reproduction, which introduces randomness to create a mixed ability of genes that are more robust. 

tf.random.set_seed(seed=2024)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(25, activation='relu',input_shape=(X_train.shape[1], )), 
    tf.keras.layers.Dropout(rate=0.2,seed=2024),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(rate=0.1,seed=2024),
    tf.keras.layers.Dense(10, activation='relu'), 
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
])

#### VANISHING GRADIENT
The vanishing gradient problem is a challenge that can occur when training deep neural networks using backpropagation: 
What it is?
The gradient, or error, calculated at the output layer of a neural network decreases in size as it's propagated back through 
the network to update weights. This can cause the initial layers of the network to learn very slowly or not at all, which can 
stall the learning process. 
Why it happens?
The vanishing gradient problem is caused by the nature of certain activation functions, such as sigmoid and hyperbolic tangent 
(tanh), and the chain rule of calculus. These functions squash input values into a limited range, and their gradients 
are small for inputs that are far from zero. 
What it means?
The vanishing gradient problem can significantly impact the performance of a model, such as its training time and prediction 
accuracy. 
How to solve it?
Regularization is a suggested approach to solve the vanishing gradient problem. 


#### EXPLODING GRADIENT problem is a common issue in deep neural networks that occurs when the gradient increases too much during 
training. This can cause the network to become unstable and unable to learn from training data. 
Why it happens?
Initial weights: The initial weights assigned to the neural network can cause large losses. 
Sharp nonlinearities: The multiplication of multiple parameters in the objective function can create sharp nonlinearities in 
parameter space. 
High learning rate: Choosing a high learning rate can cause the exploding gradient problem. 
Incorrect model architecture: Choosing an incorrect model architecture or activation function can cause the exploding gradient 
problem. 
How to solve it? 
Reducing the number of layers in the network can help, but it can also reduce the model's complexity. 
Batch normalization
Normalizing the inputs of each layer can help stabilize the learning process. 
Gradient clipping
Clipping the gradients during backpropagation can prevent them from exceeding a certain threshold. 

####INITIALIZERS
Glorot --> Tanh, Logistic, Softmax, Sigmoid
He --> ReLU
LeCun --> SELU
initializer = tf.keras.initializers.GlorotNormal(seed=None)
layer = Dense(3, kernel_initializer=initializer)

####BatchNormalization layer
Batch normalization aims to improve the training process and increase the model's generalization capability. It reduces the 
need for precise initialization of the model's weights and enables higher learning rates. That will accelerate the training 
process. Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard 
deviation close to 1.
tf.random.set_seed(seed=2024)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(25, activation='relu',input_shape=(X_train.shape[1], )), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='relu'), 
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
		

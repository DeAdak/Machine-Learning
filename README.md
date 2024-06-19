ML Algorithms types
	1. Supervised Learning
	2. Unsupervised Learning
	3. Reinforcement Learning(RL)

1. Supervised Learning
	Response
	|_Categorical(Classification)
	|	|__Single Label
	|	|	|__Binary     --> Yes/No, Male/Female
	|	|	|__Multiclass --> Red/Green/Blue/Black
	|	|
	|	|__Multi Label    --> Diabetes, Blood Pressure, Suger
	|
	|_Numerical(Regression)
	
	Model Evaluation
	|__Categorical(Classification)
	|	|__Misclassification error
	|	|__Accuracy score
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
	
	Data Pre-Processing
	|__Scaling
		|__Standard Scaler
		|__Min Max Scaler
		
	Baye's Theorem 
	(The sklearn.naive_bayes module implements Naive Bayes algorithms. These are supervised learning methods based on applying 
	Bayes’ theorem with strong (naive) feature independence assumptions.)
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
	
	Linear Model
	|
	|__Regression(Response = Numerical)
	|	|_LinearRegression
	|	|_Ridge
	|	|_Lasso
	|	|_ElasticNet
	|	|_SGDRegressor
	|	
	|__Classification(Response = Categorical)
		|_LogisticRegression --> Binary --> Sigmoid
		|					 --> Multiclass --> Softmax
		|_RidgeClassifier
		|_SGDClassifier
		
	Discriminant Analysis
	|
	|__Classification(Response = Categorical)
		|_LinearDiscriminantAnalysis
		|_QuadraticDiscriminantAnalysis
		
	Support Vector Machines
	|
	|__Regression(Response = Numerical)
	|	|_SVR(Epsilon-Support Vector Regression)
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
		|	|__AdaBoost (DecisionTree with max_depth = 1) <-- weak learner
		|	|	|__Regression(Response = Numerical)
		|	|	|	|__AdaBoostRegressor
		|	|	|
		|	|	|__Classification(Response = Categorical)
		|	|		|_AdaBoostClassifier
		|	|
		|	|__GBM
		|	|	|__Regression(Response = Numerical)
		|	|	|	|__GradientBoostingRegressor
		|	|	|
		|	|	|__Classification(Response = Categorical)
		|	|		|__GradientBoostingClassifier
		|	|
		|	|__XGBM
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
		|__Smoothing Method
		|	|__Moving Average
		|	|	|__Centered Moving Average
		|	|	|__Trailing Moving Average
		|	|
		|	|__Simple Smoothing
		|	|
		|	|__Holt's Method
		|	|	|__Linear Trend
		|	|	|__Exponential Trend
		|	|	|__Additive Damped Trend
		|	|	|__Multiplicative Damped Trend
		|	|
		|	|__Holt-Winter's Method(ExponentialSmoothing)
		|		|__Additive Trend
		|		|__Additive & Damped Trend
		|		|__Multiplicative Trend
		|		|__Additive & DampedTrend
		|__ARIMA(Auto-Regressive Integrated with Moving Average)(p,d,q)
			Auto-Regressive Model
			Moving Average Model
			ARMA(Auto-Regressive Moving Average)(p,q)
			Seasonal ARIMA(p,d,q)(P,D,Q)[S]
	
2. Unsupervised Learning
	|
	|__Cluster Analysis
	|	|__Hierarchical
	|	|	|__Agglomerative
	|	|		|__Linkage
	|	|
	|	|__Non-hierarchical
	|		|__K-Means
	|		|	|__KMeans
	|		|__DBSCAN
	|			|__DBSCAN
	|
	|__Dimensionality Reduction
	|	|__Singular Value Decomposition
	|	|__Principle Component Analysis
	|
	|__Assotion Rules Learning
	|__Recommender System

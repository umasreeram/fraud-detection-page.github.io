## Motivation

Fraud risk is everywhere. One major sector affected by fraud risk is the e-commerce industry. Online payment service companies are able to collect a vast amount of data on individuals and their transactions and use modeling to distinguish between fraudulent and non-fraudulent behaviors. In order to build our modeling skills and explore the field of fraud detection, we are applying machine learning to detect online payment frauds in a e-commerce transaction dataset from IEEE Computational Intelligence Society (IEEE-CIS) and payment service company, Vesta Corporation.

## Data & Preprocessing (Chitwan)

The (dataset)[https://www.kaggle.com/c/ieee-fraud-detection/data] provided by Vesta includes identification and transaction data on a series of online payments. The data contains the following unmasked features.

Table 1A: Unmasked Features

| Feature Name  | Description 
| ------------- | ------------- 
| TransactionDT | timedelta from a given reference datetime
| TransactionAMT | transaction payment amount in USD
| ProductCD | product code for each transaction
| addr | address of purchaser
| P_emaildomain | purchaser's email
| R_recipientdomain | recipient's email
| DeviceType | desktop or mobile
| DeviceInfo | specific machine (e.g. MacBook)

The meaning of the following features are masked but Vesta has provided the following high level descriptions about the feature categories. Note the examples below are for illustrative purposes only and these specific features may not exist in the data.

Table 1B: Masked Features

| Feature Category  | Description 
| ------------------- | ------------- 
| id12 - id38 | identies, such as network connection, digital signature, etc.
| card1 - card6 | payment card information, such as card type, card category, issue bank, etc.
| dist | distance between 2 masked locations
| C1-C14 | counting, such as how many addresses are found to be associated with the payment card, etc.
| D1-D15 | timedelta, such as days between previous transaction, etc.
| M1-M9 | match, such as names on card and address, etc.
| Vxxx | Vesta engineered rich features, including ranking, counting, and other entity relations

**Missing Data.**
All except 20 features have some missing values. We drop features where 90-100% of the values are missing. Since our dataset is so large and most of our features are masked, we decide to not pursue any complex data imputation techniques. For models that can't handle missing values such as logisitic regression, we fill NAs with 0. For models that can handle missing values such as XGBoost, we experiment with leaving missing values as is and filling missing values with -999. -999 is well outside the range of typical values and we believe that the model will be able to distinguish these values as missing and ignore them. 

**Multicollinearity**
Many of our features are derived from each other so our predictors are highly multicollinear. Because we want to extract feature importance from our models, we need to reduce multicollinearity. Since the "Vxxx" features are engineered features and not actual data, we drop one of every two highly correlated features (e.g. |Correlation coefficient| > 0.75). We drop the feature with fewer number of unqiue values, the intuition being that the feature with greater number of unqiue values contains more "granular" data. 

Although other non-Vxxx features are also multicolinear, we decide not to do drop them. The non-Vxxx features represent actual data that might be useful in distinguishing fraud vs not fraud. 

Figure 1: C features correlation matrix
[C corr plot](C_corrplot.png)

**Feature Engineering**

## Methodology (Ngan, Wendy)

- Describe how data is partioned into different segments and used to tune parameters for each model family and select best model family. Data Leakage discussion? (Ngan)
- Paramter Tuning using Bayes Search Optimization (Wendy)

## Experiments (Ngan,Uma,Aditi,Wendy)

### Approach 1: Regression Methods
We began our modeling with a simple logisitic regression model which would serve as our baseline as we explored more complex methods.

**Logistic Regression (Uma)**
- Feature selection (Chi squared/ANOVA tests)
- Dealing with class imbalance
	- random undersampling
	- clustered undersampling
- LR results w/ random undersampling on XG_LR X_1, X_2

### Approach 2: Tree Based Methods
Compared to logisitic regression, tree based methods are less susceptible to outliers and make fewer assumptions about the underlying structure of our data. So, in addition to logisitic regression, we tried tree based methods such as Random Forest, LGBM, and XGBoost. 

**Random Forest (Ngan)**
- Results for LGBM X_1, X_2, X_3, X_4 (@Ngan, I think RF don't require dummy data. Chitwan, you are correct that RF in theory should be able to handle both numerical and categorical variables. However, random forest in the context of sklearn cannot. It requires numeric data only. People recommend the implementation of H2O library which can handles numeric and categorical variables on its own wo user preprocessing step. (Ref: https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/). Unfortunately, its too late now to try it. Essentialy, they should both work, just sklearn is more computational cost but I already did that ^^!. Also, I have been using X_1 data only. Do I need to do X_2, X_3, X_4? What is the gain? On what data should we report the feature importance (A&B vs A only))? Do we need Kaggle results for all family models (LR, RF, LGBM --- I think no)? 

The most basic tree based model is Decision Tree - a single tree algorithm which is commonly refered as Classification and Regression Trees (CART). A Decision tree is a flowchart like tree structure, in which each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) represents a class label. The paths from root to leaf represent classification rules. Decision at each node is made such that it maximizes the information gain or minimizes the total entropy. The decision tree is susceptible to overfitting and hence requires pruning. However, pruned decision tree is still sentitive to high variance and instability in predicting test data. To resolve this issue, bagging decision tree model is introduced where it combines multiple decision trees. 

Random forest is identical to bagging decision tree except it adds additional randomness to the model. While splitting a node, instead of searching for the most important feature among all features, it searches for the best feature among a random subset of features. Therefore, in random forest, only a random subset of the features is taken into consideration by the algorithm for splitting a node. This results in a wide diversity that generally results in a better model.

For our case, we implemented random forest using the sklearn library "RandomForestClassifier" function. Refer to Table X for a list of hyperparameters and their corresponding search space. 

**LGBM (Aditi)**

- maybe mention sth about "Gradient-based One-Side Sampling (GOSS) to filter out the data instances for finding a split value"
- Results for LGBM X_1, X_2, X_3, X_4

**XGBoost (Wendy)**

XGBoost is a gradient boosted decision tree algorithm designed for speed and performance, that is known exceptional performance in binary classification problems with a severe class imbalance. Our XGBoost model implementation uses a histogram-based algorithm to compute the best split.

To accelerate our model training and hyperparameters tuning processes, we set up an AWS EC2 instance with GPU to train the XGBoost model on the cloud. Taking our base model as an example, this successfully decreases the training time from 58 minutes to under 3 minutes (95% decrease), and the prediction time from 2 minutes to under 8 seconds(93% decrease).

It is computationally and financially expensive to tune the hyperparameters of the XGBoost estimator. We have to priortize the parameters to be tuned (Table X) and outline a resonable search space. We also used the Bayesian optimization algorithm in the Scikit-optimize module for model tuning. After each iteration, the algorithm makes an educated guess on which set of hyperparameters is most likely to improve model performance. Therefore, this method is likely to be more efficient than other more commonly known methods, like GridSearch or random serach.

Table X. Ranked listing of XGBoost hyperparameters tuned

| Hyperparameters  | Impact on model | Importance |
| ------------- | ------------- |------------- |
| n_estimators | Number of decision trees in the model. Higher value increases complexity of the model, making the model more likely to overfit.| High|
|learning_rate| Impacts the duration needed for the model to converge and performance of the model. | High|
|max_depth| Maximum depth of each decision tree. Higher value increases complexity of the model, making the model more likely to overfit. | High|
|colsample_bytree| Number of features used by each tree. Lower value means that each tree can only consider a smaller proprtion of total columns. This avoids some columns to take too much credit for the prediction. | High|
|subsample| % of training set to subsample to build each tree. Higher value prevents overfitting, but potentially in sacrifice of performance. | Medium|
|gamma| Minimum reduction in the loss function required to make a split. Regularization parameter. Values can vary based on the loss function.| Medium|

- Results for XGBoost X_1, X_2, XG_LR X_1, X_2

## Results & Discussion 
- Table X: Model Hyperparameter and search space:
	- Logistic Regression
	- Random Forest:
		- Maximum depth of the tree: 900 - 1200
		- Number of trees in the forest: 30 - 70
		- Number of features to consider when looking for the best split: 'auto', 'log2'
			- 'auto': max_features = sqrt(n_features)
			- 'log2': max_features = log2(n_features)
	- LGBM
	- XGBoost
- Table X: Best hyperparameters for each family model and their corresponding AUC score on testing data
	- Logistic Regression
	- Random Forest:
		- Maximum depth of the tree: 32
		- Number of trees in the forest: 931
		- Number of features to consider when looking for the best split: 'log2'
		- Class weight: 'balanced' mode uses the values of training class to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
	- LGBM
	- XGBoost
- Discuss results of you experiments and which one we ended up selecting, final test AUC from Kaggle? (Wendy)
- Discuss how Vesta could operationalize this, things to consider from Uma's findings (Chitwan)




# Template info below incase you guys need it (Will not be in final report)

## Yo guys we will put our final results here!!!!

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](/hw_3_img.jpg)

```
![alt text](hw_3_img.jpg)

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/umasreeram/fraud-detection-page.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

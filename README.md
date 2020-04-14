## Motivation

Fraud risk is everywhere. One major sector affected by fraud risk is the e-commerce industry. Online payment service companies are able to collect a vast amount of data on individuals and their transactions and use modeling to distinguish between fraudulent and non-fraudulent behaviors. In order to build our modeling skills and explore the field of fraud detection, we are applying machine learning to detect online payment frauds in a e-commerce transaction dataset from IEEE Computational Intelligence Society (IEEE-CIS) and payment service company, Vesta Corporation.

## Data & Preprocessing (Chitwan)

- Add description of data (can take a lot from Kaggle post)
- Dealing with missing values
- Reducing multicollinearity
- Engineering new features

## Experiments

### Approach 1: Regression Methods
We began our modeling with a simple logisitic regression model which would serve as our baseline as we explored more complex methods.

**Logistic Regression (Uma)**
- Feature selection (Chi squared/ANOVA tests)
- Dealing with class imbalance
	- random undersampling
	- clustered undersampling
- LR results w/ random undersampling on XG_LR X_1, X_2

### Approach 2: Tree Based Methods
Compared to logisitic regression, tree based methods are less susceptible to outliers and make fewer assumptions about the underlying structure of our data. We experimented with multiple decision-tree-based ensemble methods, including Random Forest, LGBM, and XGBoost.

**Random Forest (Ngan)**
- Results for LGBM X_1, X_2, X_3, X_4 (@Ngan, I think RF don't require dummy data)

**LGBM (Aditi)**
- maybe mention sth about "Gradient-based One-Side Sampling (GOSS) to filter out the data instances for finding a split value"
- Results for LGBM X_1, X_2, X_3, X_4

**XGBoost (Wendy)**

XGBoost is a gradient boosted decision tree algorithm designed for speed and performance, that is known exceptional performance in binary classification problems with a severe class imbalance. Our XGBoost model implementation uses a histogram-based algorithm to compute the best split.

To accelerate our model training and hyperparameters tuning processes, we set up an AWS EC2 instance with GPU to train the XGBoost model on the cloud. Taking our base model as an example, this successfully decreases the training time from 58 minutes to under 3 minutes (95% decrease), and the prediction time from 2 minutes to under 8 seconds(93% decrease).

It is computationally and financially expensive to tune the hyperparameters of the XGBoost estimator. We have to priortize the parameters to be tuned (Table X) and outline a resonable search space. We also used the Bayesian optimization algorithm in the Scikit-optimize module for model tuning. After each iteration, the algorithm makes an educated guess on which set of hyperparameters is most likely to improve model performance. Therefore, this Bayesian method is likely to be more efficient than other more commonly known methods, like GridSearch or random serach.

Table X. Ranked listing of XGBoost hyperparameters tuned
| Hyperparameters  | Impact on model | Importance |
| ------------- | ------------- |------------- |
| n_estimators | Number of decision trees in the model. Too small: hinder predictive ability of the model, too large: computationally intensive, might risk overfitting| High|
|learning_rate| Too low
|max_depth| Maximum depth of each decision tree. Too low: underfitting, too high: might lead to overfitting | High|
|subsample|
|colsample_bytree|
|gamma|




- Results for XGBoost X_1, X_2, XG_LR X_1, X_2

### Results & Discussion 
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

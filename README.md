## REAL ESTATE
### PROBLEM STATEMENT
When buying a home, one of the most important things to consider is the price of the property. For real estate companies to succeed in bringing new customers, their prices must be consistent. A prediction model for the same would be a very useful tool to have, and it might also be utilized to alter property development by highlighting trademarks that enhance property values. We need to build a model to forecast housing prices.
> *Note: To start the project, `RStudio` should be installed on your PC.*

### STEP-1: Load or Read the data
Here we have taken two datasets, `housing_train.csv` and `housing_test.csv`. We need to use data in the housing_train CSV file to build a predictive model for the response variable `Price`, housing_test CSV file contains all other factors except "Price", we need to predict that using the model that we developed and save the predicted values in a CSV file.
>*Note: After loading the data, import all the required `libraries` and `Packages`.*

### STEP-2 : Data Preparation
Data preparation, also called pre-processing, is a crucial step for data analysis, which may take up to 80% of the time spent on any analytics project.

In this project, after loading the data, we need to explore & analyze the relation between each variable.
> **Tips :**
> * Check the columns that are not needed for the evaluation and then drop them or check whether any independent variable (`every variable except Price`) is not affecting the dependent variable (`Price`).
> * Check the datatype, if the datatype is other than numeric or integer, then change it.
> * If there is any interest rate or percentage value, then remove `%` and convert it to numeric.
> * If there is any category repeated, then club low-frequency categories and create dummies.
> * If there is any range present, then split the range into two numbers and take an average of it.

*Make some rough notes like below :*

```markdown
suburb       : drop it
Address      : convert to numeric and create dummies
Rooms        : numeric
Type         : convert to numeric and create dummies
Method       : convert to numeric and create dummies
SellerG      : drop it
Distance     : numeric
Postcode     : numeric
Bedroom2     : numeric
bathroom     : numeric
car          : numeric
Landsize     : numeric
BuildingArea : numeric
Yearbuilt    : drop it
CouncilArea  : convert to numeric and create dummies
```

Data preprocessing starts with the `recipe()` function which lets us define our dependent and independent variables.

***Create the recipe `dp_pipe` :***
* `update_role` : It updates the role of the variables like  `new_role="drop_vars"` and will assign a role "drop_vars" for the columns, that we want to remove. In the same way `new_role="to_dummies"` will create dummies for categorical vars.

* `step_rm` : It removes those vars that `has_role="drop_vars"`.

* `step_mutate_at` : It lets us apply a custom function.

* `step_unknown` : It lets us impute missing values with any text value.

* `step_other` : It combines infrequent categories (anything less than 2% or 0.02) of a categorical variable into a new category.

* `step_dummy` : It creates dummy variables for all the categorical variables that `has_role="to_dummies"`.

* `step_impute_median` : It imputes missing values in all numeric columns with its median.
>*Note : We have excluded `all_outcomes()` from the imputation as thats our dependent variable*.

```markdown
#create the recipe
dp_pipe= recipe(Price ~. , data= house_train) %>%
  update_role(Suburb,SellerG,YearBuilt, new_role="drop_vars") %>%
  update_role(Address, Type, Method, CouncilArea,new_role="to_dummies") %>%
  step_rm(has_role("drop_vars")) %>%
  step_mutate_at(Rooms, fn=function(x)as.numeric(as.character(x))) %>%
  step_mutate_at(Bedroom2, fn=function(x)as.numeric(as.character(x))) %>%
  step_mutate_at(Bathroom, fn=function(x)as.numeric(as.character(x))) %>%
  step_mutate_at(Car, fn=function(x)as.numeric(as.character(x))) %>%
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>%
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>%
  step_dummy(has_role("to_dummies"))  %>%
  step_impute_median(all_numeric(),-all_outcomes())
```

***Prepare the recipe***

`prep()` records everything from the train dataset, so that we can apply the same to the test dataset.

```markdown
#prepare the recipe
dp_pipe=prep(dp_pipe)
```

***Bake the recipe***

`bake()` will simply prepare train and test datasets.

```markdown
#bake the recipe
train=bake(dp_pipe, new_data = NULL)
test=bake(dp_pipe,new_data=house_test)
```

### STEP-3 : Sample the train data
We need to split the data in training and validation in the ratio of 80:20.
```markdown
s=sample(1:nrow(train),0.8*nrow(train))
t1=train[s,] #training
t2=train[-s,] #validation
```
### STEP-4 : Define the Algorithm
We have taken three supervised regression algorithms

### *Decision Tree*
To build a decision tree, we take an entire data and try to figure out how to split the data so that it makes those subsets of the data more homogeneous than before. Now we need to figure out how we define this homogeneity. We start with heterogeneous data and split that data to make it more homogeneous based on some cascading rules.

Before we take our discussion forward, let's get familiar with a little terminology. The top point where you have all your data undivided is called a parent node. subsequent partitions of your data are called nodes. Nodes that don't have any children node are called terminal nodes or leaves. Terminal nodes are the ones where the final decision is taken. Merging a node back with its parent is called pruning a tree. The number of terminal nodes is called the size of a tree.

Think about it this way without any partition in the data your best bet at prediction for a continuous
numeric target is average. Here the measure of error becomes the total sum of squares. As you partition your
data this goes down. Deviance in the case of regression trees is simply the error sum of squares. And of course, the prediction is the average of our response variable on the respective terminal node.

One very important thing that you need to notice and remember here is that the tree function can be used
for both regression and classification problems. The way it differentiates or figures out whether it needs to
build a classification or regression tree is by looking at the data type of response. If it's numeric it carries out a
regression modeling process. And if it's factor type it builds a classification model. So remember to convert
your response to factor when you are working on a classification project.

The below figure shows the best five RMSE values for Decision tree model.

![Decision_tree_best_rmse](https://github.com/samugarivishnupriya/Real-Estate/assets/85831285/a09f1751-c200-40a7-869c-aa51291b5b7e)

### *Random Forest*
Decision trees are very good at capturing non-linear patterns in the data. That's a good thing about them and
a bad thing about them as well. Good because well, they can capture non-linear patterns
very well. Bad because this capability makes them susceptible to capturing very niche patterns from the
training data which might not generalize very well. this is called overfitting or model conforming to noise in
the data.
A very simple yet powerful idea of introducing randomness in the process takes care of this problem. The name
of the algorithm that we are going to discuss is RandomForest, it works using the fact that noise is a smaller
portion of the data. If we randomly subset our data and use that to build our tree instead, likely, it will likely not be affected by noise. Let's say hypothetically 85% of the time it will not be affected by noise
and 15% of the time it will be. To counter that 15% effect, we can build many such trees, each one being
built on a different random subset of the data, and then take the average/majority vote to make a prediction.

RandomForest uses two sets of randomness to counter the effect of noise
1. Each tree is built on a random subset of observations. This averages out the effect of noisy observations.
2. For each of these trees, at each splitting node, instead of all variables in the data being used, only a
subset of variables are considered to select the splitting rules from. This averages out the effect of noisy
variables.

We'll see that R's implementation of RandomForest has 4 parameters:
 * ***mtry*** : Number of variables randomly subsetted at each node to select the splitting rule from. This
value should be an integer, greater than 1 and less than or equal to the number of predictor variables in the
data. The default value is p/3 for regression and sqrt{p} for classification problems where p is the number of
predictor variables in the data.
* ***ntree*** : This is the number of trees in the forest. There is no limit on it as such, a good starting point is
50,100 and you can try out values as large as 1000,5000. Although a very high number of trees make
sense when the data is huge as well. The default value is 500.
* ***nodesize*** : This is the minimum size of terminal nodes. This essentially stops the hairsplitting of the data,
meaning a forced split will not be considered if the node resulting from the split is too small. This
generally stems from some niche patterns in the data which do not generalize very well. Again there is
no limit on this as such but a good range to try can be between 1 to 20. The default value is 1.
* ***maxnodes*** : This controls the size of the tree, it is max number of terminal nodes. The larger you make a
tree, the more overfitting might happen. Again there is no limit on it as such but a good starting point can
be 5 and higher could be anything, although I have rarely seen people going above 100. Default is set
to NULL meaning no limits and size is controlled by other factors and data patterns themselves.

You can see that there are many possible values for these parameters. Let's see if we want to try 2 different
values for each of these parameters.
* mtry : 5, 10
* ntree : 100, 500
* maxnodes : 15, 20
* nodesize : 2, 5
This leads to 2*4 = 16 possible combinations of the parameters.

| mtry | ntree | maxnodes | nodesize |
| --- | --- | --- | --- |
| 1 | 5 | 100 | 15 | 2 |
| 2 | 10 | 100 | 15 | 2 |
| 3 | 5 | 500 | 15 | 2 |
| 4 | 10 | 500 | 15 | 2 |
| 5 | 5 | 100 | 20 | 2 |
| 6 | 10 | 100 | 20 | 2 |
| 7 | 5 | 500 | 20 | 2 |
| 8 | 10 | 500 | 20 | 2 |
| 9 | 5 | 100 | 15 | 5 |
| 10 | 10 | 100 | 15 | 5 |
| 11 | 5 | 500 | 15 | 5 |
| 12 | 10 | 500 | 15 | 5 |
| 13 | 5 | 100 | 20 | 5 |
| 14 | 10 | 100 | 20 | 5 |
| 15 | 5 | 500 | 20 | 5 |
| 16 | 10 | 500 | 20 | 5 |

We can try using all these combinations and try the resulting model performance on validation and choose the
combination which performs best.

The below figure shows the best five RMSE values for Random Forest model.

![Random_forest_best_rmse](https://github.com/samugarivishnupriya/Real-Estate/assets/85831285/57762544-27d5-4bac-b1b3-67acc5408280)

### *XGBoost*
XGBoost is an implementation of Gradient Boosted decision trees. In this algorithm, decision trees are created in sequential form. Weights play an important role in XGBoost. Weights are assigned to all the independent variables which are then fed into the decision tree which predicts results. The weight of variables predicted wrong by the tree is increased and these variables are then fed to the second decision tree. These individual classifiers/predictors then ensemble to give a strong and more precise model. It can work on regression, classification, ranking, and user-defined prediction problems.

The below figure shows the best five RMSE values for XGBoost model.

![XGboost_best_rmse](https://github.com/samugarivishnupriya/Real-Estate/assets/85831285/714d09f5-e961-43fd-9e34-9d17d5ce021e)

### STEP-5 : Prediction
Now we compare the rmse values for all the models which we have implemented.
| model | RMSE |
| --- | --- |
| Decision Tree | 357878.5 |
| Random Forest | 313790.2 |
| XGBoost | 305696.6 |

 For this project, XGBoost has given the best results of the least RMSE value of `305696.6` when compared to others. As we know less the RMSE value better the model is. We have done prediction by using `Trained XGBoost Model`.


### References
1. [https://www.geeksforgeeks.org/xgboost/](https://www.geeksforgeeks.org/xgboost/)
2. [https://drive.google.com/file/d/1nvwO4m6kOkGIg0zafzz3WS8-xJYYe0On/view?usp=sharing](https://drive.google.com/file/d/1nvwO4m6kOkGIg0zafzz3WS8-xJYYe0On/view?usp=sharing)

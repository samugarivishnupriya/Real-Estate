getwd()
#set the directory path
setwd("C:/Users/asus/Desktop/Data analyst/Projects and results of R/Project 1")

#Read the data
house_train=read.csv("housing_train.csv",stringsAsFactors = F)
house_test=read.csv("housing_test.csv",stringsAsFactors = F)

#import the library
library(dplyr)
library(stringr)
library(visdat)
library(ggplot2)
library(tidymodels)
library(car)
library(vip)
library("rpart.plot")

#function for any word or space character
Address_func=function(x){
  x=gsub("^\\w.*\\s", "", x)
  return(x)
}

house_train$Address= Address_func(house_train$Address)
house_test$Address=Address_func(house_test$Address)

vis_dat(house_train)

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

#prepare the recipe
dp_pipe=prep(dp_pipe)

#bake the recipe
train=bake(dp_pipe, new_data = NULL)

test=bake(dp_pipe,new_data=house_test)

vis_dat(train)


# PREDICTIVE MODELLING USING LINEAR REGRESSION

# Splitting the data set

set.seed(1)

s=sample(1:nrow(train),0.8*nrow(train))

t1=train[s,]

t2=train[-s,]

# Fitting Linear model to training data set

fit = lm(formula = Price ~ . -CouncilArea_Stonnington -CouncilArea_Hobsons.Bay
         -Method_X__other__ -Method_VB -Type_X__other__ -Address_Gr, data = t1)


summary(fit)

t2.pred = predict(fit , newdata=t2)

errors=t2$Price-t2.pred

rmse=errors**2 %>% mean() %>% sqrt()

Score = 212467/rmse

sort(vif(fit),decreasing = T)

fit=stats::step(fit)

summary(fit)

# Predicting test set result

y_pred = predict(fit , newdata = test)


## Decision Tree

tree_model=decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("regression")


folds = vfold_cv(train, v = 10)

tree_grid = grid_regular(cost_complexity(), 
                         tree_depth(),
                         min_n(), 
                         levels = 3)
#View(tree_grid)
doParallel::registerDoParallel()
my_res=tune_grid(
  tree_model,
  Price~.,
  resamples = folds,
  grid = tree_grid,
  metrics = metric_set(yardstick::rmse),
  control = control_grid(verbose = TRUE)
)


autoplot(my_res)+theme_light()

fold_metrics=collect_metrics(my_res)
View(fold_metrics)
x=my_res %>% show_best()
write.csv(x,'Decision_Tree_Best_roc_auc.csv',row.names = F)

final_tree_fit=tree_model %>% 
  finalize_model(select_best(my_res)) %>% 
  fit(Price~.,data=train)


# feature importance 

final_tree_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# plot the tree

rpart.plot(final_tree_fit$fit)

# predictions
train_pred=predict(final_tree_fit,new_data = train,type="prob") 
test_pred=predict(final_tree_fit,new_data = test,type="prob") 


## RANDOM FOREST
install.packages("ranger")

rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("regression") %>%
  set_engine("ranger")

folds = vfold_cv(train, v = 5)

rf_grid = grid_regular(mtry(c(5,30)), trees(c(10,500)),
                       min_n(c(2,10)),levels = 3)

doParallel::registerDoParallel()
my_res1=tune_grid(
  rf_model,
  Price~.,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(yardstick::rmse),
  control = control_grid(verbose = TRUE)
)

autoplot(my_res1)+theme_light()

fold_metrics=collect_metrics(my_res1)

y=my_res1 %>% show_best()
write.csv(y,'Random_forest_roc_auc.csv',row.names = F)

final_rf_fit=rf_model %>% 
  set_engine("ranger",importance='permutation') %>% 
  finalize_model(select_best(my_res1,"rmse")) %>% 
  fit(Price~.,data=train)

# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# predictions
train_pred=predict(final_rf_fit,new_data = train,type="prob") 
test_pred=predict(final_rf_fit,new_data = test,type="prob") 

#predict on test data
test.rf.class=predict(final_rf_fit,new_data = test)


# XGBOOST
library(xgboost)

xgb_spec = boost_tree(
  trees = 300, 
  tree_depth = tune(), 
  min_n = tune(), 
  loss_reduction = tune(),                     
  sample_size = tune(), 
  mtry = tune(),         
  learn_rate = tune(),                         
) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")


xgb_grid = grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), train),
  learn_rate(),
  size = 10
)
str(train)

xgb_grid

xgb_wf = workflow() %>%
  add_formula(Price~.) %>%
  add_model(xgb_spec)

xgb_wf

set.seed(2)
property_folds= vfold_cv(train, v=10)

set.seed(2)
xgb_res = tune_grid(
  xgb_wf,
  resamples = property_folds,
  grid = xgb_grid,
  control = control_grid(verbose = T)
)

fold_metrics1=collect_metrics(xgb_res)
write.csv(fold_metrics1,'Fold_metrics_xgboost_P1.csv',row.names = F)

z=show_best(xgb_res, "rmse")
write.csv(z,'XGBoost_best_rmse.csv',row.names = F)

best_rmse=select_best(xgb_res,"rmse")

#finalize xgboost model

final_xgb=finalize_workflow(xgb_wf,best_rmse)

#train final model using full training data
final_xgb_fit=final_xgb %>% fit(data=train) %>% extract_fit_parsnip()

final_xgb_fit %>% vip(goem="point")

#make forecast
test_forecast_xgb=predict(final_xgb_fit,new_data = test)
test_forecast_xgb
is.na(test_forecast_xgb)
write.csv(test_forecast_xgb,'Vishnupriya_samugari_P1_part2.csv',row.names = F)
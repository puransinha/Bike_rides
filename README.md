
## Bike Sharing Demand

This Python code explores several basic machine learning approaches on 
Bike Sharing Demand. 

# BUsiness Case


#### What this code does

This code allows the user to specify one one and more different 
machine learning algorithms available from the 
Python [scikit-learn](http://scikit-learn.org/stable/) library, 
to use in predicting bike demand. The user must also specificy which 
data variable(s) should be used for training, and whether to 

1. train on the training sample, in order to submit a prediction
 **OR**
2. train and test on a subset of all available data.

The first option trains the model on the full input training set. 
Depending on the choice of machine learning algorithm, 
training on the full data set might take a few minutes.

The second option splits the data (day.csv) into a training set 
(5% of the data) and a testing set (the remaining 95% of data). 
This way the accuracy of the prediction can be known immediately, 
and the bad predictions investigated. 


Finally, the user has the option to visualize the data, 
by plotting the "count" (total number of bikes rented) as a function 
of a selected data variable. 

#### How to run the code

This is a python script, which can be run by typing "Final_bikerides_assignment.py".


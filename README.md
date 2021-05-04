# Sentiment Analysis on Tripadvisor reviews
Sentiment analysis, analyzing user’s textual reviews, to understand if a comment includes a positive or negative review (i.e. binary classification model). The dataset has been scraped from the tripadvisor.it Italian web site and contains 41077 textual reviews written in the Italian language.


## Models implemented

In `tripadvisor.ipynb` different classifiers have been implemented : Decision Tree, Random Forest, Stochastic Gradient Descent (SGD) and Linear Support Vector Machine (SVM). 

To handle the imbalance of the dataset, we apply cross-validation during the training and validation steps. This approach consists of dividing all the samples in groups of subsamples. Then the prediction function is learnt using k-1 folds and the remaining fold is used for testing. The default value for the number of folds is set to 10.

## Results

Overall the best *accuracy* (**0.94**) has been obtained by applying *Linear SVM* model.

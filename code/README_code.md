
The folder PROMISE contains the .py files with which the operation and performance of the dataset PROMISE has been tested.

- *promise-test-stratified.ipynb* tests the training of the models when PROMISE dataset is stratified in 10 folds. 
- *promise-test-by-docs.ipynb* tests the training of the models when PROMISE dataset is divided by documents.

The folder ReWoRC contains the .py files with which the operation and performance of the dataset ReWoRC has been tested.

- *rework-test-stratified.ipynb* tests the training of the models when ReWoRC dataset is stratified in 10 folds.
- *rework-test-by-docs.ipynb* tests the training of the models when ReWoRC dataset is divided by documents.
- *rework-test-by-topics.ipynb* tests the training of the models when ReWoRC dataset is divided by topics.
- *rework-test-with-problem-statements.ipynb* tests again the training of the models when ReWoRC dataset is stratified in 10 folds with the difference that the statement of the problem to be solved is added to the requirements.

All experiments have been repeated 10 times (except the one where the dataset is divided by themes, which is repeated 6 times), with different datasets for training, validation and testing.

Given the high consumption of cpu and gpu resources required to run the tests, we have used the virtual environment dedicated to data science and machine learning *Kaggle*. The code is therefore adapted to run on *Kaggle* but with very few changes it can run in every Linux enviroment (directories would have to be modified, new imports would have to be included...).

The file *calculate_results.ipynb* calculates the means, standard deviations and best learning rates for the results obtained.

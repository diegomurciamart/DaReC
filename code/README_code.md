
The folder PROMISE contains the .py files with which the operation and performance of the dataset PROMISE has been tested.

- *promise-test-stratified.ipynb* tests the training of the models when PROMISE dataset is stratified in 10 folds. 
- *promise-test-by-docs.ipynb* tests the training of the models when PROMISE dataset is divided by documents.

The folder DaReC contains the .py files with which the operation and performance of the dataset DaReC has been tested.

- *darec-test-stratified.ipynb* tests the training of the models when DaReC dataset is stratified in 10 folds.
- *darec-test-by-docs.ipynb* tests the training of the models when DaReC dataset is divided by documents.
- *darec-test-by-topics.ipynb* tests the training of the models when DaReC dataset is divided by topics.
- *darec-test-with-problem-statements.ipynb* tests again the training of the models when DaReC dataset is stratified in 10 folds with the difference that the statement of the problem to be solved is added to the requirements.

All experiments have been repeated 10 times (except the one where the dataset is divided by topics, which is repeated 6 times), with different datasets for training, validation and testing.

Given the high consumption of cpu and gpu resources required to run the tests, we have used the virtual environment dedicated to data science and machine learning *Kaggle*. The code is therefore adapted to run on *Kaggle* but with very few changes it can run in every Linux enviroment (directories would have to be modified, new imports would have to be included...).

The file *calculate_results.ipynb* calculates the means, standard deviations and best learning rates for the results obtained. It also calculates t-tests to check whether variations in results using BERT, RoBERTa and DeBERTa are significant. It can be used to compare results of different experiments.

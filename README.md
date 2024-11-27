## DaReC - Dataset for Requirements Classification
___
The purpose of this github repository is to host all the work related to the final degree project *"Automatic generation of software requirements: Creation of the dataset and training of language models"*. This work includes:
  - Code
  - Datasets
  - Results
___

This project aims to create a new dataset which can be used to train LLM so the models can classify software requirements as functional or non-functional. The requirements that make up the dataset were selected from PURE: another dataset of 79 publicly available natural language requirements documents collected from the Web.

Our goal is to achieve a valid alternative to dataset PROMISE which only contains 625 requirements from 15 documents, so in the course of the project we have the performance of both in order to ultimately select which is most suitable for each case.

DaReC dataset contains 2391 requirements from 50 industrial and academical documents. We tested PROMISE and DaReC datasets in different situations and using this three language models: BERT, DeBERTa and RoBERTa

More information about the code, results and datasets can be found in the READMEs for each section.

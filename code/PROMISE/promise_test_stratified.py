
!pip install gdown -U
!pip install evaluate

import os
os.environ["WANDB_DISABLED"]="true"

!mkdir -p results_promise_docs_bert

!gdown '1zLzqY2h10TNiq9NDKl0cJBTWSfHv014Y'

"""Aqui empiezo el código igual al de colab"""

from datasets import load_dataset

FILE_REQ='/kaggle/working/NFR_PROMISE.csv'
data_req = load_dataset('csv', data_files=FILE_REQ, sep=';')
print(data_req['train']['NFR'][21])

def create_labels(row):
  return {'labels': 'F' if row['NFR'] == 'F' else 'NF'}

data_req = data_req.map(create_labels)
data_req = data_req.class_encode_column('labels')
print(set(data_req['train']['labels']))
data_req = data_req['train']
data_req.features

from transformers import AutoTokenizer
#BERT: model_name = 'google-bert/bert-base-cased'
#DEBERTA: model_name = 'microsoft/deberta-v3-base'
#ROBERTA: model_name = 'FacebookAI/roberta-base'
model_name = 'google-bert/bert-base-cased' # modificar con el modelo que queramos cada vez

tokenizer = AutoTokenizer.from_pretrained(model_name)

def encode_text(row, tokenizer):
  row_encode = tokenizer(row['Requisito'], truncation=True)
  return row_encode


encode_data = data_req.map(encode_text, fn_kwargs={'tokenizer':tokenizer})
encode_data

from sklearn.model_selection import StratifiedKFold
import numpy as np

folds = StratifiedKFold(n_splits=10, shuffle=True)

splits = folds.split(np.zeros(data_req.num_rows), encode_data["labels"])
test_sets = []
train_val_sets = []
i=1
for train_val_idxs, test_idxs in splits:
  print (i)
  print(test_idxs)
  print('*******')
  data_test = encode_data.select(test_idxs)
  data_train = encode_data.select(train_val_idxs)
  data_train_val = data_train.train_test_split(train_size=0.9, stratify_by_column='labels')
  #data_train_inner, data_val_inner = data_train.train_test_split(train_size=0.9)
  test_sets.append(data_test)
  train_val_sets.append(data_train_val)
  i = i + 1

print(test_sets[0])
print(train_val_sets[0]['test'])

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

import evaluate
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis = 1)
    return metric.compute(predictions=predictions, references=labels)

# load metric
metric_name = 'f1'
metric = evaluate.load(metric_name)

"""Si uso dos GPUs poner batch size 16, porque van 16 a cada una = 32"""

import glob
from transformers import TrainingArguments, Trainer
from sklearn.metrics import classification_report

lr_values =[2e-5, 5e-5]

for j in range(len(train_val_sets)):
    best_lr = -1
    best_f1 = -1

    # para escribir los resultados
    with open(f'results_promise_docs_bert/res_bert_{j}.txt', 'w') as f:
    # NOTA: si no va, deshacer el tab extra del siguiente bucle for
      for i, lr in enumerate(lr_values):
        print("================================================")
        print("j = " + str(j) + " // lr = " + str(lr))
        print("================================================")
      # argumentos para el entrenamiento
        training_args = TrainingArguments(
            #output_dir="my_checkpoint"+str(j),
            output_dir=f"my_checkpoint_{i}",
            overwrite_output_dir = True,
            num_train_epochs=10,
            #para probar tanto promise como nuestro coger 10 epochs
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit = 1,
            load_best_model_at_end=True,
            logging_strategy='epoch',
            optim="adamw_torch",
            per_device_train_batch_size=32,
            #lo dejaremos en 32: per_device_eval_batch_size=32,
            learning_rate=lr,
            #probar tanto 2e-5 como 5e-5
            weight_decay=0.01,
        )

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        # declarar el "entrenador"
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_val_sets[j]['train'],
            eval_dataset=train_val_sets[j]['test'],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        # realizar el entrenamiento
        trainer.train()

        out_pred = trainer.predict(train_val_sets[j]['test'])
        if out_pred.metrics['test_f1'] > best_f1:
          best_f1 = out_pred.metrics['test_f1']
          best_lr = i
      # cargar el mejor modelo y resultados para test
      best_dir = glob.glob(f"my_checkpoint_{best_lr}/checkpoint*")
      print(best_dir)
      model_best = AutoModelForSequenceClassification.from_pretrained(best_dir[0])
      trainer_best = Trainer(
            model=model_best,
            args=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

      test_data = test_sets[j]
      outputs_pred = trainer_best.predict(test_dataset=test_data)

      # model predictions
      predictions = np.argmax(outputs_pred.predictions, axis=1)

      #d_res = classification_report(test_data['labels'], predictions, digits=3, return_dict = True)
      d_res = classification_report(test_data['labels'], predictions, digits=3, output_dict=True)
      print(d_res)
      f.write("================== Best lr ================\n")
      f.write(str(best_lr))
      f.write("\n")
      f.write( "================== Resultados ================\n")
      f.write(str(d_res))

"""Aquí acaba el código de colab.
Falta hacer el zip para poder descargar los resultados (y opcionalmente usar el bot para que me avise del final de la ejecución)
"""
# ======================================================================================================
# Code executions are long, so we use a bot which sends us a message to warn that the run has ended

#import requests

# Definir la URL base y los parámetros
#url = "https://api.callmebot.com/whatsapp.php"
#params = {
#    "phone": "phonenumber",
#    "apikey": "api-key",
#    "text": "🤖¡Ejecución terminada! 🤖\nVe a echarle un vistazo ☝🤓"
#}

# Enviar la solicitud GET
#response = requests.get(url, params=params)

# Imprimir el estado de la respuesta
#print(response.status_code)
#print(response.text)
# ======================================================================================================

!zip -r results_promise_docs_bert.zip results_promise_docs_bert
from IPython.display import FileLink
FileLink(r'results_promise_docs_bert.zip')
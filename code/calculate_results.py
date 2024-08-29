import ast
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

# the directory where we can find the .txt results may change
FILE_NAME_PREFIX = ' /content/drive/MyDrive/TFG/RESULTADOS/BERT/results_bert/res_bert_'
f1_score_1 = []
precision_1 = []
recall_1 = []
f1_score_0 = []
precision_0 = []
recall_0 = []
best_lr_array = []
for i in range(10): # si hacemos la comprobación de temas "for i in range(6):"
  f_name = FILE_NAME_PREFIX+str(i)+'.txt'
  print(f_name)
  f_in = open(f_name,'r')
  lines = f_in.readlines()
  best_lr = int(lines[1])
  dict_res_string = lines[3]
  dict_res = ast.literal_eval(dict_res_string)
  f_in.close()
  f1_score_1.append(dict_res['1']['f1-score'])
  precision_1.append(dict_res['1']['precision'])
  recall_1.append(dict_res['1']['recall'])
  # para los ficheros results_promise_docs_roberta/...7 y results_promise_docs_roberta/...8 no hay ['0']
  # esto se debe a que las particiones correspondientes debían ser todo 1s
  # para debuggear esto debería repetir la ejecución o lo que voy a hacer a continuación:
  if '0' in dict_res:
    f1_score_0.append(dict_res['0']['f1-score'])
    precision_0.append(dict_res['0']['precision'])
    recall_0.append(dict_res['0']['recall'])
  best_lr_array.append(best_lr)
  print(f1_score_0)

media_f1_1 = np.mean(f1_score_1)
media_precision_1 = np.mean(precision_1)
media_recall_1 = np.mean(recall_1)
media_f1_0 = np.mean(f1_score_0)
media_precision_0 = np.mean(precision_0)
media_recall_0 = np.mean(recall_0)
print(f"La media de las f1-score de los 1s es: {media_f1_1}")
print(f"La media de las precisiones de los 1s es: {media_precision_1}")
print(f"La media de las recalls de los 1s es: {media_recall_1}")
print(f"La media de las f1-score de los 0s es: {media_f1_0}")
print(f"La media de las precisiones de los 0s es: {media_precision_0}")
print(f"La media de las recalls de los 0s es: {media_recall_0}")

std_f1_1 = np.std(f1_score_1)
std_precision_1 = np.std(precision_1)
std_recall_1 = np.std(recall_1)
std_f1_0 = np.std(f1_score_0)
std_precision_0 = np.std(precision_0)
std_recall_0 = np.std(recall_0)
print(f"La desviación estándar de las f1-score de los 1s es: {std_f1_1}")
print(f"La desviación estándar de las precisiones de los 1s es: {std_precision_1}")
print(f"La desviación estándar de las recalls de los 1s es: {std_recall_1}")
print(f"La desviación estándar de las f1-score de los 0s es: {std_f1_0}")
print(f"La desviación estándar de las precisiones de los 0s es: {std_precision_0}")
print(f"La desviación estándar de las recalls de los 0s es: {std_recall_0}")

numero_de_unos = best_lr_array.count(1)
numero_de_ceros = best_lr_array.count(0)
print(f"Es mejor escoger un learning rate de 2e-5 en {numero_de_ceros} casos.")
print(f"Es mejor escoger un learning rate de 5e-5 en {numero_de_unos} casos.")
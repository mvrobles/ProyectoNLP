
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import pandas as pd

from glob import glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange
import sys
import os

from transformers import Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM
from transformers import EarlyStoppingCallback
from transformers import Seq2SeqTrainer

import torch

import numpy as np
import pickle
import evaluate

import sacrebleu
bleu_calc = sacrebleu.BLEU()
chrf_calc = sacrebleu.CHRF(word_order=2)

def preprocess_dataset(path_dataset: str, lang_output: str):
  """
  Lee los datos y los preprocesa. Lo pasa al formato necesario DatasetDict
  y divide los datos en train, test y validación.
  Sirve para traducción de indígena a español

  input:
  - path_dataset: con la ruta en donde se encuentra la base a procesar
  - lang_output: wayuu, arh de donde va a terminar la traducción

  output:
  - dataset_dict: DatasetDict con train test y validation
  """
  # Lectura de datos y conversión a diccionario
  dataset = pd.read_csv(path_dataset)
  conv = {'esp': 'es', 'wayuu': lang_output, 'arh': lang_output}
  dataset.rename(columns = conv, inplace = True)

  dataset = [{'es': row['es'], lang_output: row[lang_output]} for _, row in dataset.iterrows()]

  # División train, test y validación
  train, test = train_test_split(dataset, test_size = 0.2, random_state = 42)
  val, test = train_test_split(test, test_size = 0.5, random_state = 42)

  # Creación de datasets
  train = Dataset.from_dict({"id": list(range(len(train))), "translation": train})
  test = Dataset.from_dict({"id": list(range(len(test))), "translation": test})
  validation = Dataset.from_dict({"id": list(range(len(val))), "translation": val})

  # Creación del diccionario
  dataset_dict = DatasetDict({"train": train, "test": test, "validation": validation})

  return dataset_dict

def tokenizar(dataset_dict, tokenizer, max_length = 150):
  """
  A partir de un DatasetDict, tokeniza los datos. Esto depende del modelo a utilizar,
  y de un modelo específico.

  input:
  - dataset_dict: con los datos de train, test y validación
  - tokenizer: tokenizer
  - max_length: de las sentencias a considerar

  output:
  - tokenized_datasets
  """

  def preprocess_function(examples):
      inputs = [ex["es"] for ex in examples["translation"]]
      targets = [ex["fi"] for ex in examples["translation"]]
      model_inputs = tokenizer(
          inputs, text_target=targets, max_length=max_length, truncation=True
      )
      return model_inputs

  # Tokenizar los datos
  tokenized_datasets = dataset_dict.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset_dict["train"].column_names,
  )

  return tokenized_datasets, tokenizer

model_path = "../results/arhuaco_reves"
eval_blues = {}

for res in glob(model_path + '/*'):
  if 'pickle' in res and 'resultados' not in res:
    with open(res, 'rb') as file:
      blue_score = pickle.load(file)['eval_bleu']
      eval_blues[res] = blue_score
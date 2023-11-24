from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import pandas as pd

def preprocess_dataset(path_dataset: str, lang_input: str):
  """
  Lee los datos y los preprocesa. Lo pasa al formato necesario DatasetDict 
  y divide los datos en train, test y validación. 
  Sirve para traducción de indígena a español

  input:
  - path_dataset: con la ruta en donde se encuentra la base a procesar
  - lang_input: wayuu, arh de donde va a comenzar la traducción

  output: 
  - dataset_dict: DatasetDict con train test y validation
  """
  # Lectura de datos y conversión a diccionario
  dataset = pd.read_csv(path_dataset)
  conv = {'esp': 'es', 'wayuu': lang_input, 'arh': lang_input}
  dataset.rename(columns = conv, inplace = True)

  dataset = [{'es': row['es'], lang_input: row[lang_input]} for _, row in dataset.iterrows()]
  
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
  
def tokenizar(dataset_dict, model_checkpoint, max_length = 150):
  """
  A partir de un DatasetDict, tokeniza los datos. Esto depende del modelo a utilizar,
  y de un modelo específico. 

  input:
  - dataset_dict: con los datos de train, test y validación
  - model_checkpoint: identificador del modelo a utilizar
  - max_length: de las sentencias a considerar

  output:
  - tokenized_datasets
  """
  # Cargar tokenizador
  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")

  def preprocess_function(examples):
      inputs = [ex["tr"] for ex in examples["translation"]]
      targets = [ex["es"] for ex in examples["translation"]]
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
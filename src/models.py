from transformers import Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer

import numpy as np
import evaluate


def get_model(tokenized_datasets, tokenizer, model_checkpoint, 
              learning_rate = 2e-5, epochs = 3, weight_decay = 0.01):

  model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
  data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
  
  metric = evaluate.load("sacrebleu")

  def compute_metrics(eval_preds):
      preds, labels = eval_preds
      # In case the model returns more than the prediction logits
      if isinstance(preds, tuple):
          preds = preds[0]

      decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

      # Replace -100s in the labels as we can't decode them
      labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
      decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

      # Some simple post-processing
      decoded_preds = [pred.strip() for pred in decoded_preds]
      decoded_labels = [[label.strip()] for label in decoded_labels]

      result = metric.compute(predictions=decoded_preds, references=decoded_labels)
      return {"bleu": result["score"]}

  args = Seq2SeqTrainingArguments(
      f"marian-finetuned-kde4-tr-to-es",
      evaluation_strategy="no",
      save_strategy="epoch",
      learning_rate = learning_rate,
      per_device_train_batch_size=32,
      per_device_eval_batch_size=64,
      weight_decay=weight_decay,
      save_total_limit=3,
      num_train_epochs=epochs,
      predict_with_generate=True,
      fp16=True,
      push_to_hub=False,
  )

  trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
  )

  return trainer
run_name = 'run_2'
output_dir = 'e5-base_' + run_name
gpuid=0
learning_rate = 5e-5
dropout = 0.5
comment = "Clipped weighted loss"
num_epochs = 10
batch_size=64
checkpoint_path=False

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    SentenceTransformerTrainingArguments,
)


import os
import sys
import wandb

import numpy as np
import torch
from torch import nn
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.trainer import WeightedSentenceTransformerTrainer
from src.loss import WeightedCosineSimilarityLoss
from src.evaluator import RocAucEvaluator

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

# === Load model ===
model_id = "intfloat/e5-base-v2"
model = SentenceTransformer(model_id)
model[0].auto_model.embeddings.dropout.p = dropout

# === Load dataset ===
dataset = load_dataset("csv", data_files="../data/train.csv")

dataset = dataset.remove_columns('id')
dataset = dataset.rename_column('question1', 'text1')
dataset = dataset.rename_column('question2', 'text2')
dataset = dataset.rename_column('target', 'label')

def prepend_query(example):
    example['text1'] = "query: " + example['text1']
    example['text2'] = "query: " + example['text2']
    return example

dataset = dataset.map(prepend_query)

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(dataset['train']['label']),
    y=dataset['train']['label']
)

split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset['train']
test_dataset = split_dataset['test']


# === Training ===

loss = WeightedCosineSimilarityLoss(
    model,
    cos_score_transformation=nn.ReLU()
)

args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    warmup_ratio=0.05,
    fp16=False,
    bf16=False,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=100,
    logging_dir="./logs",
    run_name=run_name,
    report_to="wandb",
)

dev_evaluator = RocAucEvaluator(
    sentences1=test_dataset["text1"],
    sentences2=test_dataset["text2"],
    labels=test_dataset["label"],
    batch_size=batch_size,
    clip_negative=True,
)

wandb_run_name = f"e5-base-{run_name}"
wandb.init(
    project="DuplicateSearch",
    name=wandb_run_name,
    mode='offline',
    config={
        "learning_rate": learning_rate,
        "dropout": dropout,
        "comment": comment,
    },
)

trainer = WeightedSentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=loss,
    evaluator=dev_evaluator,
    class_weights=class_weights
)

trainer.train(resume_from_checkpoint=checkpoint_path)

# === Evaluation ===
roc_auc = dev_evaluator(model)['eval_cosine_roc_auc']
print(f'ROC AUC: {roc_auc}')

wandb.log({"test/roc_auc": roc_auc})

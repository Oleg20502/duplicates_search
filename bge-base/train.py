run_name = 'run_5'
output_dir = 'bge-base_' + run_name
gpuid=1
learning_rate = 2e-5
dropout = 0.5
comment = "Clipped loss"
num_epochs = 10
batch_size=64
checkpoint_path=False




from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
)

from sentence_transformers.evaluation import (
    BinaryClassificationEvaluator,
    SimilarityFunction,
)

from transformers import AutoConfig, AutoTokenizer, AutoModel

from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import wandb

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

# === Load model ===
model_id = "BAAI/bge-base-en-v1.5"

model = SentenceTransformer(model_id)

model[0].auto_model.embeddings.dropout.p = dropout

# === Load dataset ===

dataset = load_dataset("csv", data_files="../data/train.csv")

dataset = dataset.remove_columns('id')
dataset = dataset.rename_column('question1', 'text1')
dataset = dataset.rename_column('question2', 'text2')
dataset = dataset.rename_column('target', 'label')

split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

loss = losses.CosineSimilarityLoss(
    model,
    cos_score_transformation=nn.ReLU()
)


# === Training ===

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    warmup_ratio=0.05,
    fp16=False,
    bf16=False,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=100,
    logging_dir="./logs",
    run_name=run_name,
    report_to="wandb",
)

dev_evaluator = BinaryClassificationEvaluator(
    sentences1=test_dataset["text1"],
    sentences2=test_dataset["text2"],
    labels=test_dataset["label"],
    similarity_fn_names=['cosine'],
    batch_size=batch_size,
)

wandb_run_name = f"bge-base-{run_name}_epochs-{num_epochs}"
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


trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)

trainer.train(resume_from_checkpoint=checkpoint_path)



def test_model(model, test_dataset, clip_negative=True):
    sentences1=test_dataset['text1']
    sentences2=test_dataset["text2"]
    labels=test_dataset["label"]

    sentences = sentences1 + sentences2
    embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    embeddings1 = embeddings[: len(sentences1)]
    embeddings2 = embeddings[len(sentences1) :]

    similarity = 1 - paired_cosine_distances(embeddings1, embeddings2)
    if clip_negative:
        similarity[similarity < 0] = 0.0
    return similarity


similarity = test_model(model, test_dataset)

roc_auc = roc_auc_score(test_dataset['label'], similarity)
print(f'ROC AUC: {roc_auc}')

wandb.log({"test/roc_auc": roc_auc})


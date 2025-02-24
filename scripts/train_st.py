from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

# Загружаем базовую модель
model = SentenceTransformer("intfloat/e5-base-v2")  # Можно заменить на BGE или SBERT

# Загружаем CSV-файл
df = pd.read_csv("train.csv")

# Создаем тренировочные примеры
train_samples = [InputExample(texts=[row["query1"], row["query2"]], label=float(row["label"])) for _, row in df.iterrows()]

# Создаем DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)  # Используем косинусную схожесть

# Обучаем модель
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)

# Сохраняем обученную модель
model.save("retrieval_model")


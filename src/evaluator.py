from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import roc_auc_score

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator

class RocAucEvaluator(SentenceEvaluator):
    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        labels: list[int],
        name: str = "",
        batch_size: int = 64,
        clip_negative: bool = True,
        show_progress_bar: bool = False,
    ):
        super().__init__()

        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels

        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        self.clip_negative = clip_negative

    def __call__(
        self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:

        embeddings = model.encode(
            self.sentences1 + self.sentences2,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        embeddings1 = embeddings[: len(self.sentences1)]
        embeddings2 = embeddings[len(self.sentences1) :]

        labels = np.asarray(self.labels)

        similarity = 1 - paired_cosine_distances(embeddings1, embeddings2)
        if self.clip_negative:
            similarity[similarity < 0] = 0.0

        roc_auc = roc_auc_score(labels, similarity)

        return {'eval_cosine_roc_auc': roc_auc}
        



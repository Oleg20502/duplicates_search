from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers import SentenceTransformer

class WeightedCosineSimilarityLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        cos_score_transformation: nn.Module = nn.Identity(),
    ) -> None:
        super().__init__()
        
        self.model = model
        self.cos_score_transformation = cos_score_transformation

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor,
        weights: Tensor,
        # metadata,
    ) -> Tensor:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))

        # weights = torch.tensor([m["weight"] for m in metadata], dtype=torch.float32, device=output.device)
        
        loss = (labels - output)**2 * weights
        return loss.mean()

    def get_config_dict(self) -> dict[str, Any]:
        return {"loss_fct": "Weighted MSE Loss"}

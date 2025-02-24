from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer


class WeightedSentenceTransformerTrainer(SentenceTransformerTrainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)

        if class_weights is None:
            self.class_weights = {0: 1.0, 1: 1.0}
        else:
            self.class_weights = class_weights
    
    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """
        Computes the loss for the SentenceTransformer model.

        It uses ``self.loss`` to compute the loss, which can be a single loss function or a dictionary of loss functions
        for different datasets. If the loss is a dictionary, the dataset name is expected to be passed in the inputs
        under the key "dataset_name". This is done automatically in the ``add_dataset_name_column`` method.
        Note that even if ``return_outputs = True``, the outputs will be empty, as the SentenceTransformers losses do not
        return outputs.

        Args:
            model (SentenceTransformer): The SentenceTransformer model.
            inputs (Dict[str, Union[torch.Tensor, Any]]): The input data for the model.
            return_outputs (bool, optional): Whether to return the outputs along with the loss. Defaults to False.
            num_items_in_batch (int, optional): The number of items in the batch. Defaults to None. Unused, but required by the transformers Trainer.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]: The computed loss. If `return_outputs` is True, returns a tuple of loss and outputs. Otherwise, returns only the loss.
        """
        dataset_name = inputs.pop("dataset_name", None)
        features, labels = self.collect_features(inputs)
        loss_fn = self.loss

        if isinstance(loss_fn, dict) and dataset_name:
            loss_fn = loss_fn[dataset_name]

        # Insert the wrapped (e.g. distributed or compiled) model into the loss function,
        # if the loss stores the model. Only called once per process
        if (
            model == self.model_wrapped
            and model != self.model  # Only if the model is wrapped
            and hasattr(loss_fn, "model")  # Only if the loss stores the model
            and loss_fn.model != model  # Only if the wrapped model is not already stored
        ):
            loss_fn = self.override_model_in_loss(loss_fn, model)

        # weights = inputs["weights"]
        # metadata = inputs["metadata"]
        weights = self.compute_weights(labels)
        loss = loss_fn(features, labels, weights)
        if return_outputs:
            # During prediction/evaluation, `compute_loss` will be called with `return_outputs=True`.
            # However, Sentence Transformer losses do not return outputs, so we return an empty dictionary.
            # This does not result in any problems, as the SentenceTransformerTrainingArguments sets
            # `prediction_loss_only=True` which means that the output is not used.
            return loss, {}
        return loss

    def compute_weights(self, labels):
        """
        Assign different weights based on label imbalance.
        """
        return torch.tensor(
            [self.class_weights[0] if label == 0 else self.class_weights[1] for label in labels],
            dtype=torch.float,
            device=self.model.device,
        )

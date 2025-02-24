---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:26368
- loss:CosineSimilarityLoss
widget:
- source_sentence: What is the difference between a graduate of mechanical engineering
    and a mechanic?
  sentences:
  - How do I use parental controls on Directv?
  - What is the difference between automotive and mechanical engineering?
  - What is the step response if impulse response h(t) = δ^2(t)?
- source_sentence: Are Germans proud?
  sentences:
  - When is it important to use convolutions cross channels and when is it not?
  - What are the drawbacks in voice recognition software till now that they can not
    work like talking to humans?
  - Are Germans proud to be German?
- source_sentence: How do I create an algorithm?
  sentences:
  - How do you create algorithms?
  - Should I double major in math and economics?
  - What are some trolls on Narendra Modi?
- source_sentence: Is it grammatically correct to say '' thank you for your patience
    while we worked on it.''?
  sentences:
  - What are some inspirational quotes?
  - What are the biggest problems facing Brazil?
  - Would it be grammatically correct to say "Yes it's" in place of "Yes it is"?
- source_sentence: How do I overcome smartphone addiction?
  sentences:
  - What font is used for this “confluence” graphic?
  - How should I get rid of smartphone addiction?
  - What are some tips on making it through the job interview process at Signature
    Bank?
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
- cosine_accuracy_threshold
- cosine_f1
- cosine_f1_threshold
- cosine_precision
- cosine_recall
- cosine_ap
- cosine_mcc
model-index:
- name: SentenceTransformer
  results:
  - task:
      type: binary-classification
      name: Binary Classification
    dataset:
      name: Unknown
      type: unknown
    metrics:
    - type: cosine_accuracy
      value: 0.8617747440273038
      name: Cosine Accuracy
    - type: cosine_accuracy_threshold
      value: 0.64251708984375
      name: Cosine Accuracy Threshold
    - type: cosine_f1
      value: 0.8060876410390974
      name: Cosine F1
    - type: cosine_f1_threshold
      value: 0.5695635676383972
      name: Cosine F1 Threshold
    - type: cosine_precision
      value: 0.7555336940482046
      name: Cosine Precision
    - type: cosine_recall
      value: 0.8638920134983127
      name: Cosine Recall
    - type: cosine_ap
      value: 0.8473315293266096
      name: Cosine Ap
    - type: cosine_mcc
      value: 0.700999757659813
      name: Cosine Mcc
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained on the csv dataset. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - csv
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'How do I overcome smartphone addiction?',
    'How should I get rid of smartphone addiction?',
    'What font is used for this “confluence” graphic?',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Binary Classification

* Evaluated with [<code>BinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.BinaryClassificationEvaluator)

| Metric                    | Value      |
|:--------------------------|:-----------|
| cosine_accuracy           | 0.8618     |
| cosine_accuracy_threshold | 0.6425     |
| cosine_f1                 | 0.8061     |
| cosine_f1_threshold       | 0.5696     |
| cosine_precision          | 0.7555     |
| cosine_recall             | 0.8639     |
| **cosine_ap**             | **0.8473** |
| cosine_mcc                | 0.701      |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### csv

* Dataset: csv
* Size: 26,368 training samples
* Columns: <code>text1</code>, <code>text2</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | text1                                                                             | text2                                                                              | label                                           |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:------------------------------------------------|
  | type    | string                                                                            | string                                                                             | int                                             |
  | details | <ul><li>min: 3 tokens</li><li>mean: 15.26 tokens</li><li>max: 54 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 15.94 tokens</li><li>max: 120 tokens</li></ul> | <ul><li>0: ~67.30%</li><li>1: ~32.70%</li></ul> |
* Samples:
  | text1                                                                | text2                                                             | label          |
  |:---------------------------------------------------------------------|:------------------------------------------------------------------|:---------------|
  | <code>How can I get someone's email from their Twitter ID?</code>    | <code>How do I find out someone’s email from Twitter ID?</code>   | <code>0</code> |
  | <code>Where are some nice places to visit in Berlin, Germany?</code> | <code>What are some interesting places to visit in Berlin?</code> | <code>1</code> |
  | <code>What is a beta tester?</code>                                  | <code>What was it like to be a beta tester of Quora?</code>       | <code>0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Evaluation Dataset

#### csv

* Dataset: csv
* Size: 26,368 evaluation samples
* Columns: <code>text1</code>, <code>text2</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | text1                                                                             | text2                                                                              | label                                           |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:------------------------------------------------|
  | type    | string                                                                            | string                                                                             | int                                             |
  | details | <ul><li>min: 3 tokens</li><li>mean: 15.72 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 15.84 tokens</li><li>max: 276 tokens</li></ul> | <ul><li>0: ~67.30%</li><li>1: ~32.70%</li></ul> |
* Samples:
  | text1                                                                                                                      | text2                                                                                | label          |
  |:---------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------|
  | <code>What are beginner programmers interested in?</code>                                                                  | <code>What are some small, but challenging programs for beginner programmers?</code> | <code>0</code> |
  | <code>What is the scope of construction project management in NICMAR for fresher? Which one is best RICS or NICMAR?</code> | <code>Which is better: RICS or NICMAR?</code>                                        | <code>0</code> |
  | <code>What specifically defines the entrepreneurial spirit?</code>                                                         | <code>What is entrepreneurial?</code>                                                | <code>0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `learning_rate`: 2e-05
- `num_train_epochs`: 5
- `warmup_ratio`: 0.1

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step | Training Loss | Validation Loss | cosine_ap |
|:------:|:----:|:-------------:|:---------------:|:---------:|
| 0.3030 | 100  | 0.3642        | 0.1858          | 0.7404    |
| 0.6061 | 200  | 0.1473        | 0.1435          | 0.8118    |
| 0.9091 | 300  | 0.128         | 0.1369          | 0.8300    |
| 1.2121 | 400  | 0.1142        | 0.1226          | 0.8351    |
| 1.5152 | 500  | 0.1067        | 0.1206          | 0.8449    |
| 1.8182 | 600  | 0.1016        | 0.1161          | 0.8500    |
| 2.1212 | 700  | 0.0918        | 0.1156          | 0.8432    |
| 2.4242 | 800  | 0.0829        | 0.1140          | 0.8434    |
| 2.7273 | 900  | 0.0793        | 0.1142          | 0.8528    |
| 3.0303 | 1000 | 0.0806        | 0.1119          | 0.8507    |
| 3.3333 | 1100 | 0.068         | 0.1106          | 0.8506    |
| 3.6364 | 1200 | 0.0681        | 0.1099          | 0.8520    |
| 3.9394 | 1300 | 0.0657        | 0.1098          | 0.8495    |
| 4.2424 | 1400 | 0.0618        | 0.1097          | 0.8511    |
| 4.5455 | 1500 | 0.0594        | 0.1108          | 0.8490    |
| 4.8485 | 1600 | 0.0598        | 0.1104          | 0.8473    |


### Framework Versions
- Python: 3.12.8
- Sentence Transformers: 3.4.1
- Transformers: 4.48.2
- PyTorch: 2.5.1+cu124
- Accelerate: 1.3.0
- Datasets: 3.2.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
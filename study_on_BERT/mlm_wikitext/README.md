

```bash 
python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```

```python
ModelArguments(
    model_name_or_path="roberta-base",
    model_type=None,
    config_overrides=None,
    config_name=None,
    tokenizer_name=None,
    cache_dir=None,
    use_fast_tokenizer=True,
    model_revision="main",
    use_auth_token=False,
)
DataTrainingArguments(
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    train_file=None,
    validation_file=None,
    overwrite_cache=False,
    validation_split_percentage=5,
    max_seq_length=None,
    preprocessing_num_workers=None,
    mlm_probability=0.15,
    line_by_line=False,
    pad_to_max_length=False,
    max_train_samples=None,
    max_eval_samples=None,
)
```

`wikitext` dataset looks as following.
```python
DatasetDict({
    test: Dataset({
        features: ['text'],
        num_rows: 4358
    })
    train: Dataset({
        features: ['text'],
        num_rows: 36718
    })
    validation: Dataset({
        features: ['text'],
        num_rows: 3760
    })
})
```


`tokenizer`
``` python
PreTrainedTokenizerFast(
    name_or_path="roberta-base",
    vocab_size=50265,
    model_max_len=512,
    is_fast=True,
    padding_side="right",
    truncation_side="right",
    special_tokens={
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "sep_token": "</s>",
        "pad_token": "<pad>",
        "cls_token": "<s>",
        "mask_token": AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=False),
    },
)
```


`sample tokenized data`
```python
{'input_ids': [0, 2], 'attention_mask': [1, 1], 'special_tokens_mask': [1, 1]}
```

https://huggingface.co/docs/transformers/v4.19.0/en/main_classes/trainer#transformers.TrainingArguments
- do_train: default false
- do_eval: default false
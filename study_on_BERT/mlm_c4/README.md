## DataCollator
https://github.com/huggingface/transformers/blob/v4.19.0/src/transformers/data/data_collator.py#L75

https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling

```
python dataload.py \
    --model_name_or_path roberta-base \
    --dataset_name allenai/c4 \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```
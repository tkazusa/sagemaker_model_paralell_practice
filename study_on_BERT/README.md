# BERT Pre Training on C4 Japanese
## requirements.txt
- sagemaker>=2.48.0

## Run the training script on SageMaker training jos

```bash
 $ python create_training_job.py
```
Hyperparameters passed to training job are defined in `config.json` as followings, 

```JSON
    "hyperparameters": {
        "model_name_or_path": "roberta-base",
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-2-raw-v1",
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "do_train": true,
        "do_eval": true,
        "output_dir": "/opt/ml/model"
    }
```

Another hyperparemeters are defined in `ModelArguments` class, `DataTrainingArguments` class and [`TrainingArguments`](https://huggingface.co/docs/transformers/v4.19.0/en/main_classes/trainer#transformers.TrainingArguments) class in `run_mlm.py`.

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
## Dataset

We use `wikitext` dataset and it looks as follows.

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
```
= Valkyria Chronicles III = \n', '', ' Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . 
```


## Tokenizer

We use pretrained `roberta-base` tokenizer.

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

Tokenized sample data looks follows.
```bash
Tokenized dataset sample:  [0, 2, 0, 5457, 468, 44068, 6374, 41674, 6395, 5457, 1437, 50118, 2, 0, 2, 0, 2211, 267, 38183, 117, 468, 44068, 6374, 155, 4832, 1890, 36902, 41674, 36, 2898, 4832, 47416, 23133, 18164, 42393, 21402, 20024, 48018, 50033, 49080, 49587, 49432, 48947, 49017, 246, 2156, 6474, 479, 468, 44068, 6374, 9, 5, 36954, 155, 4839, 2156, 10266, 4997, 7, 25, 468, 44068, 6374, 41674, 6395, 751, 1429, 2156, 16, 10, 15714, 774, 787, 12, 1039, 816, 569, 177, 2226, 30, 43561, 8, 2454, 4, 36753, 13, 5, 15592, 39435, 479, 30939, 11, 644, 1466, 11, 1429, 2156, 24, 16, 5, 371, 177, 11, 5, 468, 44068, 6374, 651, 479, 23564, 154, 5, 276, 24904, 9, 15714, 8, 588, 787, 12, 1039, 86, 23841, 25, 63, 20193, 2156, 5, 527, 1237, 12980, 7, 5, 78, 177, 8, 3905, 5, 22, 8603, 13802, 22, 2156, 10, 14914, 831, 1933, 2754, 5, 1226, 9, 7155, 493, 148, 5, 4665, 5122, 12560, 1771, 54, 3008, 3556, 909, 1414, 8, 32, 30259, 136, 5, 16659, 1933, 22, 2912, 424, 415, 219, 22546, 22, 479, 1437, 50118, 2, 0, 20, 177, 880, 709, 11, 1824, 2156, 3406, 81, 10, 739, 4745, 9, 5, 173, 626, 15, 468, 44068, 6374, 41674, 3082, 479, 616, 24, 12544, 5, 2526]
```

We can decode tokenized data with `tokenizer.decode()`. 

```
Decoded sample:  <s></s><s> = Valkyria Chronicles III = 
</s><s></s><s> Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3, lit. Valkyria of the Battlefield 3 ), commonly referred to as Valkyria Chronicles III outside Japan, is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable. Released in January 2011 in Japan, it is the third game in the Valkyria series. Employing the same fusion of tactical and real @-@ time gameplay as its predecessors, the story runs parallel to the first game and follows the " Nameless ", a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven ". 
</s><s> 
```

# Language model training examples

This repository is for examples to learn how to train a language model with HuggingFace(Jax/Flax background). See details on [the original repo](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling).

## Setup
### Launch an EC2 instance 
- Type: g4dn.12xlarge
- AMI: Deep Learning AMI (Ubuntu 18.04) Version 59.0
- Volume Size: > 2TB

### Requirements

Follow the [guide](https://github.com/huggingface/transformers/tree/main/examples/flax) for setup to install JAX on GPUs.
- Python >=3.8
- CUDA?
- CuDNN?


### Installation

To install python libraries, run 
```bash
$ pip install -r requirements.txt
```

To install [GitLFS](https://github.com/git-lfs/git-lfs/), run
```bash
$ sudo apt install git-lfs
$ git lfs install
```

### Login to HuggingFace.

If you are required to login HuggingFace repository, run
```bash
$ huggingface-cli login
```
 
## Train Tokenizer
The toknizer used for the benchmark is a customized sentencepies unigram tokenizer which is heavily inspired from [yandex-research/DeDLOC's tokenizer model](https://github.com/yandex-research/DeDLOC/blob/5c994bc64e573702a9a79add3ecd68b38f14b548/sahajbert/tokenizer/tokenizer_model.py). The tokenizer is trained on the complete Norwegian dataset of [OSCAR](https://huggingface.co/datasets/oscar). 

```bash
$ python train_tokenier.py
```

## Train masked language model.

```bash 
$ python run_t5_mlm_flax.py \
	--output_dir="./norwegian-t5-base/output" \
	--model_type="t5" \
	--config_name="./norwegian-t5-base" \
	--tokenizer_name="./norwegian-t5-base" \
	--dataset_name="oscar" \
	--dataset_config_name="unshuffled_deduplicated_no" \
	--max_seq_length="512" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--adafactor \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="10000" \
	--eval_steps="2500" \
	--push_to_hub
```

### Changes from the original script

- Changed `--output_dir` command line argument from `./norwegian-t5-base` to `./norwegian-t5-base/output`


- Removed `use_auth_token` argument on [L719](https://github.com/huggingface/transformers/blob/7783fa6bb3dca3aa10283bd7f382d224615e44c6/examples/flax/language-modeling/run_t5_mlm_flax.py#L719) because it caused `an unexpected keyword augument`error. 

```bash
    model = FlaxT5ForConditionalGeneration(
        config,
        seed=training_args.seed,
        dtype=getattr(jnp, model_args.dtype),
        # use_auth_token=True if model_args.use_auth_token else None,
    )
```

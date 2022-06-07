# Train the RoBERTa using SageMaker Model Parallelism, the training compiler, and FSx for Lustre with C4/en dataset.

## Setup FSx for Lustre
TBD

## Setup dataset
### GitLFS 
```
$ sudo apt install git-lfs
```

### Download c4/en 305GiB
```
$ GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
$ git lfs pull --include "en/*"
```

## Reference
- [RoBERTa/BERT/DistilBERT and masked language modeling](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling)
- [Compile and Train the GPT2 Model using the Transformers Trainer API with the SST2 Dataset for Single-Node Multi-GPU Training](https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-training-compiler/huggingface/pytorch_multiple_gpu_single_node/language-modeling-multi-gpu-single-node.ipynb)

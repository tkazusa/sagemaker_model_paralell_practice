import os

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer


def batch_iterator(batch_size: int = 1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]


if __name__ == "__main__":
    # load dataset, total size 6.5 GB
    dataset = load_dataset("oscar", "unshuffled_deduplicated_no", split="train")

    # Instantiate tokenizer
    tokenizer = ByteLevelBPETokenizer()
    # 　Special Tokens used in ByteLevelBPETokenizer
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    # Customized training
    tokenizer.train_from_iterator(
        iterator=batch_iterator(),
        vocab_size=50265,
        min_frequency=2,
        special_tokens=special_tokens,
    )
    # print(tokenizer.all_special_tokens)
    # print(tokenizer.all_special_ids)

    # Create a directory to store the tokenizer file
    data_dir = "./norwegian-roberta-base"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # Save files to disk
    tokenizer.save(data_dir + "/tokenizer.json")

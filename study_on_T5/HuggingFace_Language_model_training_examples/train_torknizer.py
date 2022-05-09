import os

from datasets import load_dataset
from transformers import T5Config

from tokenizer_model.t5_tokenizer_model import SentencePieceUnigramTokenizer


# Build an iterator over this dataset
def batch_iterator(input_sentence_size=None):
    if input_sentence_size is None:
        input_sentence_size = len(dataset)
    batch_length = 100
    for i in range(0, input_sentence_size, batch_length):
        yield dataset[i : i + batch_length]["text"]


if __name__ == "__main__":
    # load dataset, total size 6.5 GB
    dataset = load_dataset("oscar", "unshuffled_deduplicated_no", split="train")

    # Instantiate tokenizer
    tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")
    # Customized training
    vocab_size = 32_000
    input_sentence_size = None

    tokenizer.train_from_iterator(
        iterator=batch_iterator(input_sentence_size=input_sentence_size), vocab_size=vocab_size, show_progress=True
    )
    # print(tokenizer.all_special_tokens)
    # print(tokenizer.all_special_ids)

    # Create a directory to store the tokenizer file
    data_dir = "./norwegian-t5-base"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # Save files to disk
    tokenizer.save(data_dir + "/tokenizer.json")
    config = T5Config.from_pretrained("google/t5-v1_1-base", vocab_size=tokenizer.get_vocab_size())
    config.save_pretrained("./norwegian-t5-base")

import glob

from datasets import load_dataset
from transformers import GPT2TokenizerFast

DATA_DIR = "c4_processed/"

if __name__ == "__main__":
    """
    train_data_files = glob.glob(DATA_DIR+"/c4-train.*")
    validation_data_files = glob.glob(DATA_DIR+"/c4-validation.*")
    data_files = {"train": train_data_files, "validation": validation_data_files}
    dataset = load_dataset('json', data_files=data_files, cache_dir=model_args.cache_dir)
    """

    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/downloads/extracted/f6855d14eabd1dffc5d358a0ad139f54dc8037faf196eb3eb37a12a669a1582a",
            "validation": "data/downloads/extracted/f6855d14eabd1dffc5d358a0ad139f54dc8037faf196eb3eb37a12a669a1582a"
            }
        )

    print(dataset)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Tokenize dataset and truncate the max length by 512
    dataset = dataset.map(lambda e: tokenizer(e['text'], max_length=512, truncation=True), num_proc=96)
    print(dataset)
    dataset = dataset.filter(lambda e: len(e['input_ids']) >= 512, num_proc=96)
    print(dataset)
    dataset = dataset.remove_columns('text')
    
    # shuffled_dataset = dataset.shuffle(seed=42)
    # dataset=shuffled_dataset.train_test_split(test_size=0.1)
    train_dataset = dataset['train'].shuffle(seed=42)
    validation_dataset = dataset['validation']

    # Write the processed dataset into files
    # Specify your own path to save the files
    validation_path = DATA_DIR + "/validation"
    train_path = DATA_DIR + "/train"

    num_shards = 1024
    for i in range(0, num_shards):
        name=f"{train_path}/train_dataset_512_filtered_{i}"
        print(name)
        shard=train_dataset.shard(num_shards=num_shards, index=i)
        print(shard)
        shard.to_json(f"{name}.json", orient="records", lines=True)

    num_shards = 8
    for i in range(0, num_shards):
        name=f"{validation_path}/validation_dataset_512_filtered_{i}"
        print(name)
        shard_validation = validation_dataset.shard(num_shards=num_shards, index=i)
        print(shard_validation)
        shard_validation.to_json(f"{name}.json", orient="records", lines=True)


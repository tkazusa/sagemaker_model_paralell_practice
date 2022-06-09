import glob

from datasets import load_dataset
from transformers import GPT2TokenizerFast

RAW_DATA_DIR = "en_unzip"
SAVE_DIR = "en_gpt_preprocessed_small"

if __name__ == "__main__":
    """
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/downloads/extracted/f6855d14eabd1dffc5d358a0ad139f54dc8037faf196eb3eb37a12a669a1582a",
            "validation": "data/downloads/extracted/f6855d14eabd1dffc5d358a0ad139f54dc8037faf196eb3eb37a12a669a1582a"
            }
        )
    """
   
    # train_data_files = glob.glob(DATA_DIR+"/c4-train.*")
    # validation_data_files = glob.glob(DATA_DIR+"/c4-validation.*")
    
    train_data_files = glob.glob(RAW_DATA_DIR+"/c4-train.00000-of-01024.json")
    validation_data_files = glob.glob(RAW_DATA_DIR+"/c4-validation.00000-of-00008.json")
    
    for train_data_file in train_data_files:
        dataset = load_dataset('json', data_files=[train_data_file], cache_dir="./data/cache")
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
        dataset = dataset.shuffle(seed=42)
        
        train_path = SAVE_DIR + "/train"
        save_path=f"{train_path}/train_dataset_512_filtered_{train_data_file[9:]}"
        dataset.to_json(save_path, orient="records", lines=True)

    for validation_data_file in validation_data_files:
        dataset = load_dataset('json', data_files=[validation_data_file], cache_dir="./data/cache")
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
        validation_path = SAVE_DIR + "/validation"
        save_path=f"{validation_path}/validation_dataset_512_filtered_{validation_data_file[14:]}"
        dataset.to_json(save_path, orient="records", lines=True)    

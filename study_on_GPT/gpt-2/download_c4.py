import glob

from datasets import load_dataset

c4_subset = load_dataset('allenai/c4', data_files='en/c4-train.00001-of-01024.json.gz', cache_dir='./data')
print(c4_subset)
print(c4_subset["train"]["text"][0:10])

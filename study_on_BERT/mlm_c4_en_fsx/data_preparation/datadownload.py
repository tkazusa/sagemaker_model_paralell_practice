import glob

from datasets import load_dataset

c4_subset = load_dataset('allenai/c4', data_files='en/c4-train.0000*-of-01024.json.gz', cache_dir= './data')

data_files = glob.glob("./data_mid/downloads/extracted/*")
print(data_files)

c4_subset = load_dataset('json', data_files=data_files)
print(c4_subset)
print(c4_subset["train"]["text"][0:10])


from glob import glob
import os

files = glob("data/crypto_processed/*usd.csv") + glob("data/crypto_processed/*ust.csv")

for file in files:
    basename = os.path.basename(file)

    os.rename(file, os.path.join("data/crypto_processed_2", basename))
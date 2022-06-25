from time import sleep
import numpy as np
import pandas as pd
import argparse

from tradegame import TradeActionException, TradeGame
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="path of dataset")
args = parser.parse_args()

df = pd.read_csv(args.dataset)

trainer = Trainer()
trainer._df_bro = df

trainer.run()
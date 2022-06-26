import argparse
import operator
import pandas as pd
import pygad
import torch
from tradegame import TradeActionException

from trainer import model

from tradegame import TradeGame

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="checkpoint path")
parser.add_argument("--test", help="test path")

args = parser.parse_args()
ga_instance : pygad.GA = pygad.load(args.path)

df = pd.read_csv(args.test)

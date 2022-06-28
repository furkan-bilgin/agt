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


for best in ga_instance.best_solutions:
    df = pd.read_csv(args.test)

    tg = TradeGame(df)

    def predictor(inputs):
        inputs = torch.tensor([inputs], dtype=torch.float32)

        return pygad.torchga.predict(model=model,
                            solution=best,
                            data=inputs)

    while not tg.is_done():
        try:
            tg.step(predictor)
        except TradeActionException:
            pass

    print(tg.fitness())
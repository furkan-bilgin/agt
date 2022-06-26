import os
from time import sleep
import numpy as np
import pandas as pd
import argparse
from glob import glob
from tradegame import TradeActionException, TradeGame
from trainer import Trainer
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", help="path of datasets")
parser.add_argument("--gencount", help="number of generations", type=int)
parser.add_argument("--checkpointpath", help="path of checkpoint")
args = parser.parse_args()

if __name__ == "__main__":
    full_df = None
    df_list = []
    
    def next_df():
        if len(df_list) > 0:
            return df_list.pop(0)
        
        full_df = pd.read_csv(data_list.pop(0))
        groups = full_df.groupby(np.arange(len(full_df.index)) // 1000)
        for (_, frame) in groups:
            df_list.append(frame)

        return next_df()

    def gen_callback(ga_instance):
        generation = ga_instance.generations_completed
        fitness = ga_instance.best_solution()[1]
        
        pbar.update(1)
        tqdm.write(f"Fitness = {fitness}")
        if generation % 100 == 0:
            trainer.set_df(next_df())
        if generation % 500 == 0:
            trainer.ga_instance.save(os.path.join(ga_save_path, f"checkpoint.gen.{generation}"))
            tqdm.write(f"Saved checkpoint in gen {generation}")
            

    data_list = glob(os.path.join(args.datapath, "*"))
    
    trainer = Trainer(gen_callback=gen_callback, gen_count=args.gencount)
    pbar = tqdm(total=args.gencount)
    ga_save_path = args.checkpointpath

    trainer.set_df(next_df())

    trainer.run()
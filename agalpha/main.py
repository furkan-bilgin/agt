from concurrent.futures import process
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
parser.add_argument("--processcount", help="number of processes while training", type=int)
parser.add_argument("--checkpointpath", help="path of checkpoint")
args = parser.parse_args()

SPLIT_COEFFICIENT = 750
SAVE_CHECKPOINT_GENERATIONS = 300
CHANGE_DATASET_GENERATIONS = 100

if __name__ == "__main__":
    df_list = []
    
    def next_df():
        if len(df_list) > 0:
            n = df_list.pop(0)
            if len(n.index) < SPLIT_COEFFICIENT / 2:
                return next_df()
            return n
        
        full_df = pd.read_csv(data_list.pop(0))
        groups = full_df.groupby(np.arange(len(full_df.index)) // SPLIT_COEFFICIENT)
        for (_, frame) in groups:
            df_list.append(frame)

        return next_df()

    def gen_callback(ga_instance):
        generation = ga_instance.generations_completed
        fitness = ga_instance.best_solution()[1]
        
        pbar.update(1)
        tqdm.write(f"Fitness = {fitness}")
        if generation % CHANGE_DATASET_GENERATIONS == 0:
            trainer.set_df(next_df())
            trainer.save_best_solution()

        if generation % SAVE_CHECKPOINT_GENERATIONS == 0:
            trainer.ga_instance.save(os.path.join(ga_save_path, f"checkpoint.gen.{generation}"))
            tqdm.write(f"Saved checkpoint in gen {generation}")
            

    data_list = glob(os.path.join(args.datapath, "*"))
    
    trainer = Trainer(gen_callback=gen_callback, gen_count=args.gencount, process_count=args.processcount)
    pbar = tqdm(total=args.gencount)
    ga_save_path = args.checkpointpath

    trainer.set_df(next_df())
    trainer.run()
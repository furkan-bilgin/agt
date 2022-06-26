from multiprocessing import Pool
import multiprocessing
import signal
import numpy as np
from pandas import DataFrame
import torch
import pygad.torchga as torchga
import pygad

from tradegame import TradeActionException, TradeGame

class PooledGA(pygad.GA):
    def cal_pop_fitness(self):
        global fitness_wrapper
        pop_fitness = self.pool.map(fitness_wrapper, list(zip(self.population, [self.df] * len(self.population))))

        pop_fitness = np.array(pop_fitness)
        return pop_fitness


model = torch.nn.Sequential(torch.nn.Linear(22, 75),
                                torch.nn.Tanh(),
                                torch.nn.Linear(75, 75),
                                torch.nn.Tanh(),
                                torch.nn.Linear(75, 75),
                                torch.nn.Tanh(),
                                torch.nn.Linear(75, 13),
                                torch.nn.Tanh())
current_df = None

def fitness_wrapper(data):
    global current_df
    solution, df = data

    current_df = df
    return fitness_func(solution, 0)
    
def fitness_func(solution, solution_index):
    global model, current_df

    def predictor(inputs):
        inputs = torch.tensor([inputs], dtype=torch.float32)

        return pygad.torchga.predict(model=model,
                            solution=solution,
                            data=inputs)
    
    tg = TradeGame(current_df)

    while not tg.is_done():
        try:
            tg.step(predictor)
        except TradeActionException as e:
            pass

    return tg.fitness() 

class Trainer:
    def __init__(self, gen_callback=None, gen_count=1000):
        self._init_ga(gen_count)
        self.gen_callback = gen_callback
        
    def _init_ga(self, gen_count):
        global fitness_func, fitness_wrapper, model
        def callback_generation(ga_instance):
            #print("Generation = {generation}".format(generation=ga_instance.generations_completed))
            #print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
            if self.gen_callback is not None:
                self.gen_callback(ga_instance)

        torch_ga = torchga.TorchGA(model=model,
                           num_solutions=20)
                           
        num_generations = gen_count # Number of generations.
        num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
        initial_population = torch_ga.population_weights # Initial population of network weights

        self.ga_instance = PooledGA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating,
                            initial_population=initial_population,
                            fitness_func=fitness_func,
                            on_generation=callback_generation)
    def set_df(self, df):
        self.ga_instance.df = df

    def run(self):
        global ignore_signals
        with Pool(processes=10, initializer=ignore_signals) as pool:
            try:
                self.ga_instance.pool = pool
                self.ga_instance.run()
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
#pipenv run python main.py --dataset data/crypto_processed/albt-usd.csv

def ignore_signals():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
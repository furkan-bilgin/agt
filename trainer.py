import numpy as np
from pandas import DataFrame
import torch
import pygad.torchga as torchga
import pygad

from tradegame import TradeActionException, TradeGame

class PooledGA(pygad.GA):
    def cal_pop_fitness(self):
        pop_fitness = self.pool.map(self.fitness_wrapper, self.population)

        pop_fitness = np.array(pop_fitness)
        return pop_fitness

class Trainer:
    def __init__(self):
        self._init_model()
        self._init_ga()

    def _init_model(self):
        self.model = torch.nn.Sequential(torch.nn.Linear(22, 75),
                                torch.nn.Tanh(),
                                torch.nn.Linear(75, 75),
                                torch.nn.Tanh(),
                                torch.nn.Linear(75, 75),
                                torch.nn.Tanh(),
                                torch.nn.Linear(75, 13),
                                torch.nn.Tanh())

    def _init_ga(self):
        def fitness_func(solution, solution_index):
            def predictor(inputs):
                inputs = torch.tensor([inputs], dtype=torch.float32)

                return pygad.torchga.predict(model=self.model,
                                    solution=solution,
                                    data=inputs)
            
            tg = TradeGame(self.get_current_df())

            while not tg.is_done():
                try:
                    tg.step(predictor)
                except TradeActionException as e:
                    #print(e)
                    pass

            print("done")
            return tg.fitness()

        def callback_generation(ga_instance):
            print("Generation = {generation}".format(generation=ga_instance.generations_completed))
            print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


        torch_ga = torchga.TorchGA(model=self.model,
                           num_solutions=25)
                           
        num_generations = 250 # Number of generations.
        num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
        initial_population = torch_ga.population_weights # Initial population of network weights

        self.ga_instance = PooledGA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating,
                            initial_population=initial_population,
                            fitness_func=fitness_func,
                            on_generation=callback_generation)

        self.ga_instance.fitness_wrapper = lambda sol : fitness_func(sol, 0)

    def get_current_df(self) -> DataFrame:
        return self._df_bro

    def run(self):
        self.ga_instance.run()
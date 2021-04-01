import multiprocessing as mp
import sys
from datetime import date
from functools import partial
from random import choices, randint, randrange, random
from typing import List, Callable, Tuple, Any

import numpy as np
import pandas as pd
import pandas_datareader as web
from dateutil.relativedelta import relativedelta

np.set_printoptions(threshold=sys.maxsize, suppress=True)


def POGA(symbols=List[str], population_size: int = 32, generation_limit: int = 10_000, risk_limit: float = 0.99, target: float = 20.
         , resample: str = '5D', std_sample: int = 5):

    # creating the genetic evolutionary algo
    Genome = List[Any]  # list of the possible portfolio weights -> float between 0-100
    Population = List[Genome]  # list of different weight groupings ie populations -> list of genomes
    FitnessFunc = Callable[[Genome], float]  # int is the fitness value
    PopulateFunc = Callable[[], Population]  # takes in nothing and returns new solutions
    SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]  # takes population and fitness func to
    # select two solutions to be the parents of the next generation
    CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]  # takes two genomes and returns two new ones
    MutationFunc = Callable[[Genome], Genome]  # takes in a genome and sometimes mutates and returns a new one

    def generate_genome(
            length=len(symbols)) -> Genome:  # length = the number of stocks were inputting -> len(df) or len(symbols)
        genome_weights = []
        w_list = np.arange(0, 1.001, 0.001, dtype=float).flatten()  # list of potential weights in given range
        w = np.random.choice(w_list, length, replace=True)  # picks random weights from list to be used for stocks
        weights = np.divide(w, np.sum(w))  # list of weights that add to 1

        return weights

    def generate_population(size: int = population_size, genome_length: int = len(
        symbols)) -> Population:  # this function makes our population the desired size
        return [generate_genome(genome_length) for _ in range(size)]

    def get_stock_data():
        # getting all required stock data -> expected returns and volatility / risk
        SOURCE = 'yahoo'
        END = date.today()
        START = np.subtract(END, relativedelta(years=4))
        df = web.DataReader(symbols, data_source=SOURCE, start=START, end=END)['Adj Close']
        df['Time Stamp'] = date.today()
        df.to_csv('POGA_stock_data.csv')

    #get_stock_data() # un commment this to gather needed stock data

    def fitness(genome: Genome, risk_limit: float = risk_limit, things=symbols) -> float:  # things = list of stocks
        if len(genome) != len(things):  # sending error if lengths arent equal
            raise ValueError('genome and things must be the same length')

        df = pd.read_csv('POGA_stock_data.csv', index_col=['Date'])
        df = df.iloc[:, :-1]
        df.index = pd.DatetimeIndex(df.index)

        if list(symbols) != list(df.columns.values): # refreshes the data if the inputs are different than the current data
            print('Getting new stock data...')
            get_stock_data()
            sys.exit('New data collected - re run the program for up to date results')


        NUM_TRADING_DAYS_PER_YEAR = 252  # the average number of trading days a year
        #individual_er = df.resample(resample).last().pct_change().mean()  # individual earning for stocks

        
        weight_vec = (np.indices((len(df.resample(resample).last()), len(df.columns)))[0] + 1) ** 2
        weighted = weight_vec / np.sum(weight_vec) # sums to 1
        pct_change = df.resample(resample).last().pct_change().fillna(0).values
        weighted_sum = []
        for i in range(len(df.columns)):
            x = np.multiply(weighted[:, i], pct_change[:, i])
            weighted_sum.append(np.sum(x))
        individual_er = pd.DataFrame(data=[weighted_sum], columns=df.columns)


        covariance_matrix = df.pct_change().apply(lambda x: np.log(1 + x)).cov()  # cov matrix of all stocks in symbols
        variance = covariance_matrix.mul(genome, axis=0).mul(genome, axis=1).sum().sum()  # portfolio variance
        std = np.sqrt(variance)  # daily standard deviation
        ann_std = np.multiply(std, (np.sqrt(std_sample)))  # port volatility = annual standard deviation

        # calculating composite performance ratios -> alpha, sharpe
        risk = 0
        returns = 0

        for i, thing in enumerate(things):  # i = number index and thing = name of stock
            individual_ret = np.multiply(individual_er[thing].values, genome[i])  # return % for each stock in list of things

            if genome[i] != 0.:
                returns += individual_ret

        risk += ann_std  # portfolio risk

        rr = returns / risk

        if risk > risk_limit:  # if the risk is over the limit
            return 0, risk, returns  # return 0 as its not a solution
        else:
            return rr, risk, returns  # if the risk isnt over limit then return the returns from that portfolio

    def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
        return choices(
            population=population,
            weights=[fitness_func(genome)[2] for genome in population],
            # assigns the fitness from above to values so that
            # the most fit weights (ones with best returns) are more likely to be chosen
            k=2
        )

    def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
        if len(a) != len(b):
            raise ValueError('Genomes a and b must be of same length')

        length = len(a)
        if length < 2:
            return a, b

        p = randint(1, length - 1)
        offspring_1 = np.concatenate((a[0:p], b[p:]))
        offspring_1 = offspring_1 / np.sum(offspring_1)

        offspring_2 = np.concatenate((b[0:p], a[p:]))
        offspring_2 = offspring_2 / np.sum(offspring_2)

        return offspring_1, offspring_2  # returns a tuple of ndarrays containing the offspring of the parents

    def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
        for _ in range(num):
            index = randrange(len(genome))
            # genome[index] = genome[index] if random() > probability else (genome[index] - 1) and (genome / np.sum(genome))
            # genome[index] = genome[index] if random() > probability else np.random.shuffle(genome)
            if random() > probability:
                genome[index] = genome[index]
            else:
                genome[index] = np.add(genome[index], 1)
                genome = np.divide(genome, np.sum(genome))

        return genome

    def run_evolution(
            populate_func: PopulateFunc,
            fitness_func: FitnessFunc,
            fitness_limit: float,  # end condition; if this lim is met then it ends and goal met
            selection_func: SelectionFunc = selection_pair,
            crossover_func: CrossoverFunc = single_point_crossover,
            mutation_func: MutationFunc = mutation,
            generation_limit: int = generation_limit  # if fitness lim not met this is how long it will run
    ) -> Tuple[Population, int]:
        population = populate_func()  # gets the first generation

        for i in range(generation_limit):
            population = sorted(population, key=lambda genome: fitness_func(genome)[2], reverse=True)

            info = [fitness_func(genome) for genome in population]
            print('Gen:', i + 1) #, population) # un comment this to get the printed weights
            print(f'Population R/R: {[list(pos[0]) for pos in info]}')
            print(f'Population returns: {[list(pos[2]) for pos in info]}')
            print(f'Population risk: {[list(pos[1]) for pos in info]}')

            if (i + 1) % 2 == 0:
                POGA_output = pd.DataFrame(columns=['Weights', 'Proj Return', 'Risk'])
                POGA_output['Weights'] = pd.Series(population)
                POGA_output['Proj Return'] = pd.Series([pos[2] for pos in info])
                POGA_output['Risk'] = pd.Series([pos[1] for pos in info])
                POGA_output['R/R'] = pd.Series([pos[0] for pos in info])
                POGA_output['Symbols'] = pd.Series(list(symbols))
                POGA_output.to_csv('POGA_output.csv', mode='a')

            if fitness_func(population[0])[2] >= fitness_limit:
                break

            next_generation = population[0:2]  # keep top 2 results for next gen

            for z in range(int(len(population) / 2) - 1):  # generating the next generation in addition to the top two
                parents = selection_func(population, fitness_func)
                offspring_a, offspring_b = crossover_func(parents[0], parents[1])

                sum_a = np.nansum(offspring_a)
                offspring_a = np.nan_to_num(offspring_a, copy=False, nan=(1 - sum_a))
                offspring_a = offspring_a / np.sum(offspring_a)

                sum_b = np.nansum(offspring_b)
                offspring_b = np.nan_to_num(offspring_b, copy=False, nan=(1 - sum_b))
                offspring_b = offspring_b / np.sum(offspring_b)

                offspring_a = mutation_func(offspring_a) # i changed from -> mutation_func(offspring_a)
                sum_a = np.nansum(offspring_a)
                offspring_a = np.nan_to_num(offspring_a, copy=False, nan=(1 - sum_a))
                offspring_a = offspring_a / np.sum(offspring_a)

                offspring_b = mutation_func(offspring_b) # changed this too like above one
                sum_b = np.nansum(offspring_b)
                offspring_b = np.nan_to_num(offspring_b, copy=False, nan=(1 - sum_b))
                offspring_b = offspring_b / np.sum(offspring_b)


                next_generation += [offspring_a, offspring_b]

            population = next_generation
            print()
        population = sorted(population, key=lambda genome: fitness_func(genome)[2], reverse=True)

        return population, i, fitness_func #, index  # i is how many generations were ran

    population, generations, fitness_func = run_evolution(
        populate_func=partial(
            generate_population, size=population_size, genome_length=len(symbols)
        ),
        fitness_func=partial(
            fitness, things=symbols, risk_limit=risk_limit,
        ),
        fitness_limit=target,
        generation_limit=generation_limit
    )

    print(f'number of generations performed: {generations + 1}')
    print(f'best solution: {population[0]}')
    print(f'Return: {(fitness_func(population[0])[2] * 100)} %')
    print(f'Risk: {(fitness_func(population[0])[1] * 100)} %')


"""symbols = pd.read_csv('top_25_6mo_appreciations.csv')
symbols = [vals[0] for vals in symbols.values]"""
symbols = ['AAPL', 'TSLA', 'SPY', 'GRWG', 'GTYH']


processes = []

for x in range(2):
    p = mp.Process(target=POGA, args=(symbols,))
    if __name__ == '__main__':
        p.start()
        processes.append(p)

for p in processes:
    p.join()


#POGA(symbols)
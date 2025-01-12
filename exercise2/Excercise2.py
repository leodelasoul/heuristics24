import logging
import os
from functools import partial
import random
from itertools import cycle
from typing import List, Callable, Any
import time
from pymhlib.demos.common import run_optimization
from pymhlib.gvns import GVNS
from pymhlib.log import init_logger
from pymhlib.population import Population
from pymhlib.scheduler import Method, Scheduler, Result, MethodStatistics
from pymhlib.settings import parse_settings, settings, get_settings_parser, get_settings_as_str, OwnSettings
from pymhlib.solution import Solution
from pymhlib.ssga import SteadyStateGeneticAlgorithm

from exercise1.v2_MWCCPInstance import v2_MWCCPInstance
from MWCCPSolutionEGA import MWCCPSolutionEGA

DIRNAME = os.path.dirname(__file__)

FILENAME: str = os.path.join(DIRNAME, '../competition_instances/inst_500_40_00021')
FILENAME1: str = os.path.join(DIRNAME, '../test_instances/medium_large/inst_500_40_00001')
FILENAME_MED: str = os.path.join(DIRNAME, '../test_instances/medium/inst_200_20_00001')
FILENAME_LARGE: str = os.path.join(DIRNAME, '../test_instances/medium_large/inst_500_40_00001')
FILENAME_LARGE1: str = os.path.join(DIRNAME, '../test_instances/large/inst_1000_60_00001')

class MyPopulation(Population):
    parameter_tune = None
    def __new__(cls, sol: Solution, meths_ch: List[Method], own_settings: dict = None):
        """Create population of mh_pop_size solutions using the list of construction heuristics if given.

        If sol is None or no constructors are given, the population is initialized empty.
        sol itself is just used as template for obtaining further solutions.
        """
        own_settings = OwnSettings(own_settings) if own_settings else settings
        size = own_settings.mh_pop_size
        obj = super(Population, cls).__new__(cls, size, Solution)
        obj.own_settings = own_settings
        if sol is not None and meths_ch:
            # cycle through construction heuristics to generate population
            # perform all construction heuristics, take best solution
            meths_cycle = cycle(meths_ch)
            idx = 0
            while idx < size:
                m = next(meths_cycle)
                sol = sol.copy()
                res = Result()
                sol.x = m.func(sol, m.par, res)
                if own_settings.mh_pop_dupelim and obj.duplicates_of(sol) != []:
                    continue  # do not add this duplicate
                obj[idx] = sol
                if res.terminate:
                    break
                idx += 1
        return obj

    def roulette_wheel_selection(self) -> int:
        total_fitness = sum(individual.obj() for individual in self)
        pick = random.uniform(0, total_fitness)
        current = 0
        for idx, individual in enumerate(self):
            current += individual.obj()
            if current >= pick:
                return idx
        return len(self) - 1

    def select(self) -> int:
        """Select one solution and return its index.

        So far calls tournament_selection. May be extended in the future.
        """
        try:
            if self.parameter_tune["mh_selection_method"] == "roulette":
                return self.roulette_wheel_selection()
            else:
                return self.tournament_selection()
        except:
            return self.roulette_wheel_selection()

class GeneticAlgorithm(Scheduler):

    def __init__(self, sol: Solution, meths_ch: List[Method],
                 meth_cx: Callable[[Solution, Solution], Solution],
                 meth_mu: Method,
                 meth_li: Method,
                 own_settings: dict = None):
        """Initialization.

        :param sol: solution to be improved
        :param meths_ch: list of construction heuristic methods
        :param meth_cx: a crossover method
        :param meth_mu: a mutation method
        :param meth_li: an optional local improvement method
        :param own_settings: optional dictionary with specific settings
        """
        population = MyPopulation(sol, meths_ch, own_settings)
        super().__init__(sol, meths_ch + [meth_mu] + [meth_li], own_settings, population=population)
        self.method_stats["cx"] = MethodStatistics()
        self.meth_cx = meth_cx
        self.meth_mu = meth_mu
        self.meth_ls = meth_li
        self.own_setting = own_settings
        self.incumbent = self.population[random.randint(0,self.population.size)-1]

    def run(self, own_settings=None):
        population = self.population
        population.parameter_tune = own_settings
        while True:
            # create a new solution
            p1 = population[population.select()].copy()

            # methods to perform in this iteration
            methods: List[Method] = []
            # optional crossover
            if random.random() < self.own_settings.mh_ssga_cross_prob:
                p2 = population[population.best()].copy()

                # workaround for Method not allowing a second Solution as parameter
                def meth_cx(crossover, par2: Solution, par1: Solution, _par: Any, _res: Result):
                    crossover(par1, par2, _par)

                meth_cx_with_p2_bound = partial(meth_cx, self.meth_cx, p2)

                meth = Method("crossover", meth_cx_with_p2_bound, own_settings["mh_fixed_crossover"] if own_settings is not None else None)
                methods.append(meth)

            # mutation
            methods.append(self.meth_mu)
            # optionally local search
            if self.meth_ls and random.random() < self.own_settings.mh_ssga_loc_prob:
                methods.append(self.meth_ls)

            res = self.perform_methods(methods, p1)

            if res.terminate:
                break

            # Replace worst individual in population
            worst = population.worst()
            population[worst].copy_from(p1)

            # Update best solution
            if p1.is_better(self.incumbent):
                # self.incumbent = p1
                self.incumbent.copy_from(p1)

    def perform_methods(self, methods: List[Method], sol: Solution) -> Result:
        res = Result()
        obj_old = sol.obj()
        method_name = ""
        for method in methods:
            if method_name != "":
                method_name += "+"
            method_name += method.name

            method.func(sol, method.par, res)
            if res.terminate:
                break
        t_end = time.process_time()

        self.iteration += 1
        new_incumbent = self.update_incumbent(sol, t_end - self.time_start)
        terminate = self.check_termination()
        self.log_iteration(method_name, obj_old, sol, new_incumbent, terminate, res.log_info)
        if terminate:
            self.run_time = time.process_time() - self.time_start
            res.terminate = True

        return res


if __name__ == '__main__':
    parser = get_settings_parser()
    parser.add_argument("--alg", type=str, default='ssga', help='ssga, ssga_tuned')
    parser.add_argument("--inst_file", type=str, default=FILENAME1,
                        help='problem instance file')
    parser.add_argument("--meths_ch", type=int, default=1,
                        help='number of construction heuristics to be used')
    parser.add_argument("--meths_li", type=int, default=1,
                        help='number of local improvement methods to be used')
    parser.add_argument("--meths_sh", type=int, default=5,
                        help='number of shaking methods to be used')
    parser.add_argument("--meths_de", type=int, default=3,
                        help='number of destroy methods to be used')
    parser.add_argument("--meths_re", type=int, default=3,
                        help='number of repair methods to be used')


    parse_settings(None, 0)

    init_logger()

    logger = logging.getLogger("pymhlib")
    logger.info("pymhlib demo for solving MWCCP")

    ###INIT
    tuning_File = "../tuning_instances/small/inst_50_4_00001"
    tuning_File1 = "../tuning_instances/medium/inst_200_20_00001"
    tuning_File2 = "../tuning_instances/medium_large/inst_500_40_00002.txt"
    mWCCPInstance = v2_MWCCPInstance(FILENAME1)
    mWCCPSolution = MWCCPSolutionEGA(mWCCPInstance)

    ### Parameter
    # best parameters from tuning: 0.8 , selection method:  roulette , mh_pop_size:  500 , mh_titer:  1000
    parser.set_defaults(mh_ttime=5000) # time limit
    parser.set_defaults(alg="ssga")
    parser = get_settings_parser()
    settings.mh_pop_size = 500 #Init population size
    settings.mh_pop_dupelim = False # Allow duplicates
    settings.mh_ssga_cross_prob = 1 # whether to use crossover , kinda useless
    settings.mh_ssga_loc_prob = 0.1 # whether to use local search
    settings.mh_titer = 1000
    alg = GeneticAlgorithm(mWCCPSolution,
                           [Method("construct heu{i}", mWCCPSolution.construct, i) for i in range(settings.meths_ch)],
                           mWCCPSolution.crossover,
                           Method("mutation", mWCCPSolution.shaking, 1),
                           Method("local_search", mWCCPSolution.local_improve, 1))

    if settings.alg == 'ssga':
        output_file = "test_compare_ga/medium_large/inst_500_40_00001.csv"
        with open(output_file, "w") as file:
            file.write("ACO\tGA\n")

        for i in range(3): #number for test runs
            alg.run({
                "mh_fixed_crossover": 0.8,
                "mh_selection_method": "roulette"})
            logger.info("")
            alg.main_results()

            with open(output_file, "a") as file:  # Append to the file
                file.write(f"{0}\t{alg.incumbent.obj()}\n")

            alg = GeneticAlgorithm(mWCCPSolution,  # reinit the alg
                                   [Method("construct heu{i}", mWCCPSolution.construct, i) for i in
                                    range(settings.meths_ch)],
                                   mWCCPSolution.crossover,
                                   Method("mutation", mWCCPSolution.shaking, 1),
                                   Method("local_search", mWCCPSolution.local_improve, 1))


    if settings.alg == 'ssga_tuned':
        tuned_parameter = {
            "mh_fixed_crossover": [0.6, 0.8, 0.9],
            "mh_selection_method": ["tournament", "roulette"],
            "mh_pop_size": [100, 500],
            "mh_titer": [100, 1000]
        }
        count = 0
        for i in range(3): #permutate
            for j in range(2):
                count += 1
                settings.mh_pop_size = tuned_parameter["mh_pop_size"][j]
                settings.mh_titer = tuned_parameter["mh_titer"][j]
                alg.run({
                    "mh_fixed_crossover": tuned_parameter["mh_fixed_crossover"][i],
                    "mh_selection_method": tuned_parameter["mh_selection_method"][j],
                })
                print("best solution: ", alg.incumbent, " with obj value: ", alg.incumbent.obj())
                print(f"time spent: {alg.run_time:.3f} seconds")
                print("paramter selection for run:", count )
                print("fixed crossoverpoint: ", tuned_parameter["mh_fixed_crossover"][i],
                      ", selection method: ", tuned_parameter["mh_selection_method"][j],
                      ", mh_pop_size: ", tuned_parameter["mh_pop_size"][j],
                      ", mh_titer: ", tuned_parameter["mh_titer"][j],
                )
                print("---------------------")

                alg = GeneticAlgorithm(mWCCPSolution,  #reinit the alg
                                       [Method("construct heu{i}", mWCCPSolution.construct, i) for i in
                                        range(settings.meths_ch)],
                                       mWCCPSolution.crossover,
                                       Method("mutation", mWCCPSolution.shaking, 1),
                                       Method("local_search", mWCCPSolution.local_improve, 1))

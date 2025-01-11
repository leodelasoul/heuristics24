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
FILENAME: str = os.path.join(DIRNAME, '../test_instances/small/inst_50_4_00001')

class MyPopulation(Population):
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

        self.incumbent = self.population[random.randint(0,self.population.size)]

    def run(self):
        population = self.population
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
                    crossover(par1, par2)

                meth_cx_with_p2_bound = partial(meth_cx, self.meth_cx, p2)

                meth = Method("crossover", meth_cx_with_p2_bound, None)
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
                self.incumbent = p1


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
    parser.set_defaults(mh_titer=100) # number of iterations
    parser.set_defaults(mh_ttime=180) # time limit

    parser = get_settings_parser()
    parser.add_argument("--alg", type=str, default='ssga', help='optimization algorithm to be used')
    parser.add_argument("--inst_file", type=str, default=FILENAME,
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
    logger.info("pymhlib demo for solving %s", "MWCCP")

    ###INIT
    mWCCPInstance = v2_MWCCPInstance(FILENAME)
    mWCCPSolution = MWCCPSolutionEGA(mWCCPInstance)
    settings.mh_pop_size = 1000 #Init population size
    settings.mh_pop_dupelim = False
    settings.mh_ssga_cross_prob = 1
    settings.mh_ssga_loc_prob = 0.1

    if settings.alg == 'ssga':
        alg = GeneticAlgorithm(mWCCPSolution,
                               [Method("construct heu{i}", mWCCPSolution.construct, i) for i in range(settings.meths_ch)],
                               mWCCPSolution.crossover,
                               Method("mutation", mWCCPSolution.shaking, 1),
                               Method("local_search", mWCCPSolution.local_improve, 1))
        alg.run()
        logger.info("")
        alg.main_results()
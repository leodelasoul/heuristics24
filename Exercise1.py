import os

from pymhlib.settings import get_settings_parser

import MWCCPInstance
import MWCCPSolution
from pymhlib.demos.common import run_optimization, data_dir
from pymhlib.settings import get_settings_parser
import os

#imports from common.py
import logging
from typing import Callable

from pkg_resources import resource_filename

from pymhlib.alns import ALNS
from pymhlib.gvns import GVNS
from pymhlib.log import init_logger
from pymhlib.par_alns import ParallelALNS
from pymhlib.pbig import PBIG
from pymhlib.sa import SA
from pymhlib.scheduler import Method
from pymhlib.settings import parse_settings, settings, get_settings_parser, get_settings_as_str
from pymhlib.solution import Solution
from pymhlib.ssga import SteadyStateGeneticAlgorithm


DIRNAME = os.path.dirname(__file__)
FILENAME: str = os.path.join(DIRNAME, 'test_instances/small/inst_50_4_00001')

if __name__ == '__main__':
    parser = get_settings_parser()
    parser.set_defaults(mh_titer=1000)
    ###INIT
    mWCCPInstance = MWCCPInstance.MWCCPInstance("instance")  # FILENAME
    mWCCPInstance.set_problem_instance()
    mWCCPSolution = MWCCPSolution.MWCCPSolution(mWCCPInstance, len(mWCCPInstance.get_instance()["u"]))
    mWCCPInstance.get_instance()
    mWCCPInstance.draw_instance()

    print(mWCCPSolution.calc_objective())


    # from common.py file
    """
    init_logger()
    logger = logging.getLogger("pymhlib")
    logger.info("pymhlib demo for solving %s", "MWCCP")
    logger.info(get_settings_as_str())
    instance = mWCCPInstance(settings.inst_file)
    logger.info("%s instance read:\n%s", "MWCCP", str(instance))
    solution = mWCCPSolution(instance)

    logger.info("Solution: %s, obj=%f\n", solution, solution.obj())

    # solution.initialize(0)

    """

    #TODO: find out how settings are used in common.py so that following code works
    alg = GVNS(mWCCPSolution,
                   [Method(f"ch{i}", MWCCPSolution.construct, i) for i in range(settings.meths_ch)],
                   [],
                   [],
                   None)
    alg.run()
    #logger.info("")
    alg.method_statistics()
    alg.main_results()

    # #run_optimization('MWCCP', MWCCPInstance, MWCCPSolution, "instance")





import os

from pymhlib.settings import get_settings_parser

import util
from MWCCPInstance import *
from MWCCPSolution import *
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
FILENAME_COMPET: str = os.path.join(DIRNAME, 'competition_instances/inst_50_4_00001')
#FILENAME_LARGE: str = os.path.join(DIRNAME, 'test_instances/la')
if __name__ == '__main__':
    parser = get_settings_parser()
    parser.set_defaults(mh_titer=1000)
    ###INIT
    #mWCCPInstance = MWCCPInstance(FILENAME)  # FILENAME
    mWCCPInstance = MWCCPInstance("instance")  # FILENAME
    mWCCPInstance.set_problem_instance()
    mWCCPSolution = MWCCPSolution(mWCCPInstance)
    #mWCCPSolution = MWCCPSolution.MWCCPSolution(mWCCPInstance, len(mWCCPInstance.get_instance()["u"]))
    mWCCPInstance.get_instance()




    parser = get_settings_parser()
    parser.add_argument("--alg", type=str, default='gvns', help='optimization algorithm to be used '
                                                                '(gvns, alns, pbig, par_alns, ssga, sa)')
    parser.add_argument("--inst_file", type=str, default=mWCCPInstance,
                        help='problem instance file')
    parser.add_argument("--meths_ch", type=int, default=1,
                        help='number of construction heuristics to be used')
    parser.add_argument("--meths_li", type=int, default=1,
                        help='number of local improvement methods to be used')
    parser.add_argument("--meths_sh", type=int, default=5,
                        help='number of shaking methods to be used')
    parse_settings(None, 0)

    # from common.py file

    init_logger()
    logger = logging.getLogger("pymhlib")
    logger.info("pymhlib demo for solving %s", "MWCCP")
    logger.info(get_settings_as_str())
    #instance = mWCCPInstance(settings.inst_file)
    logger.info("%s instance read:\n%s", "MWCCP", str(mWCCPInstance))
    #solution = mWCCPSolution(instance)
    util.text = "Before"
    #logger.info("Solution: %s, obj=%f\n", MWCCPSolution, MWCCPSolution.obj())
    util.draw_instance(u=mWCCPSolution.instance_u, x=mWCCPSolution.x, w=mWCCPSolution.instance_w)


    # #TODO: find out how settings are used in common.py so that following code works
    alg = GVNS(mWCCPSolution,
                    [Method(f"construct{i}",  MWCCPSolution.construct_random, i) for i in range(settings.meths_ch)],
                    [Method(f"local-2opt{i}", MWCCPSolution.local_improve, i) for i in range(1, settings.meths_li + 1)],
                    [],
                    None, False)
    alg.run()
    logger.info("")
    alg.method_statistics()
    alg.main_results()
    util.text = "After"
    util.draw_instance(u=mWCCPSolution.instance_u, x=mWCCPSolution.x, w=mWCCPSolution.instance_w)


    # #run_optimization('MWCCP', MWCCPInstance, MWCCPSolution, "instance")


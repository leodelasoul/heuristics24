import logging
import os

from pymhlib.demos.common import run_optimization
from pymhlib.gvns import GVNS
from pymhlib.log import init_logger
from pymhlib.scheduler import Method
from pymhlib.settings import parse_settings, settings, get_settings_parser, get_settings_as_str
from pymhlib.ssga import SteadyStateGeneticAlgorithm

from exercise1.v2_MWCCPInstance import v2_MWCCPInstance
from MWCCPSolutionEGA import MWCCPSolutionEGA

DIRNAME = os.path.dirname(__file__)
FILENAME: str = os.path.join(DIRNAME, '../test_instances/small/inst_50_4_00001')

if __name__ == '__main__':
    parser = get_settings_parser()
    parser.set_defaults(mh_titer=10) # number of iterations
    parser.set_defaults(mh_ttime=180) # time limit

    parser = get_settings_parser()
    parser.add_argument("--alg", type=str, default='ssga', help='optimization algorithm to be used '
                                                                '(gvns, alns, pbig, par_alns, ssga, sa)')
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
    logger.info(get_settings_as_str())
    logger.info("%s instance read:\n%s", "MWCCP", str(v2_MWCCPInstance))

    ###INIT
    mWCCPInstance = v2_MWCCPInstance(FILENAME)
    mWCCPSolution = MWCCPSolutionEGA(mWCCPInstance)

    if settings.alg == 'ssga':
        alg = SteadyStateGeneticAlgorithm(mWCCPSolution,
                                          [Method("ch{i}", mWCCPSolution.construct, i) for i in
                                           range(settings.meths_ch)],
                                          mWCCPSolution.crossover,
                                          Method("mu", mWCCPSolution.shaking, 1),
                                          Method("ls", mWCCPSolution.local_improve, 1),
                                          None)



    #
    # if settings.alg == 'ssga':
    # alg = SteadyStateGeneticAlgorithm(solution,
    #                                   [Method("ch{i}", mWCCPSolution.construct, i) for i in
    #                                    range(settings.meths_ch)],
    #                                   mWCCPSolution.crossover,
    #                                   Method("mu", mWCCPSolution.shaking, 1),
    #                                   Method("ls", mWCCPSolution.local_improve, 1),
    #                                   own_settings)
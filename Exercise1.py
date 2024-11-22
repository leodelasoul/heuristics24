import logging
import os

from pymhlib.gvns import GVNS
from pymhlib.log import init_logger
from pymhlib.scheduler import Method
from pymhlib.settings import parse_settings, settings, get_settings_parser, get_settings_as_str

from MWCCPInstance import *
from MWCCPSolution import *

DIRNAME = os.path.dirname(__file__)
FILENAME: str = os.path.join(DIRNAME, 'test_instances/small/inst_50_4_00001')
FILENAME_COMPET: str = os.path.join(DIRNAME, 'competition_instances/inst_50_4_00001')
# FILENAME_LARGE: str = os.path.join(DIRNAME, 'test_instances/la')
if __name__ == '__main__':
    parser = get_settings_parser()
    parser.set_defaults(mh_titer=100)
    ###INIT
    mWCCPInstance = MWCCPInstance(FILENAME_COMPET)  # FILENAME
    mWCCPInstance.set_problem_instance()
    mWCCPSolution = MWCCPSolution(mWCCPInstance)
    mWCCPInstance.get_instance()
    ###Parser arguments
    parser.add_argument("--alg", type=str, default='gvns', help='optimization algorithm to be used '
                                                                '(gvns, alns, pbig, par_alns, ssga, sa)')
    parser.add_argument("--inst_file", type=str, default=mWCCPInstance,
                        help='problem instance file')
    parser.add_argument("--meths_ch", type=int, default=5,
                        help='number of construction heuristics to be used')
    parser.add_argument("--meths_li", type=int, default=1,
                        help='number of local improvement methods to be used')
    parser.add_argument("--meths_sh", type=int, default=5,
                        help='number of shaking methods to be used')
    parse_settings(None, 0)

    init_logger()
    logger = logging.getLogger("pymhlib")
    logger.info("pymhlib demo for solving %s", "MWCCP")
    logger.info(get_settings_as_str())
    logger.info("%s instance read:\n%s", "MWCCP", str(mWCCPInstance))
    util.text = "Initial probleminstance"

    util.draw_instance(u=mWCCPSolution.instance_u, x=mWCCPSolution.x, w=mWCCPSolution.instance_w)
    #
    # alg = GVNS(mWCCPSolution,
    #            [Method(f"construct{i}", MWCCPSolution.construct, i) for i in range(settings.meths_ch)],
    #            [Method(f"local-2opt{i}", MWCCPSolution.local_improve, i) for i in range(1, settings.meths_li + 1)],
    #            [Method(f"sh{i}", MWCCPSolution.shaking, i) for i in range(1, settings.meths_sh + 1)],
    #            None, False)
    #
    alg = GVNS(mWCCPSolution,
               [Method(f"construct{i}", MWCCPSolution.construct, i) for i in range(settings.meths_ch)],
               [Method(f"local-2opt{i}", MWCCPSolution.local_improve, i) for i in range(1, settings.meths_li + 1)],
               [],
               None, False)

    alg.run()
    logger.info("")
    alg.method_statistics()
    alg.main_results()
    util.text = "Applied heuristic optimization methods"
    util.draw_instance(u=mWCCPSolution.instance_u, x=mWCCPSolution.x, w=mWCCPSolution.instance_w)


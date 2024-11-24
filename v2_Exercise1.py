import logging
import os

from pymhlib.gvns import GVNS
from pymhlib.log import init_logger
from pymhlib.scheduler import Method
from pymhlib.settings import parse_settings, settings, get_settings_parser, get_settings_as_str

from v2_MWCCPInstance import *
from v2_MWCCPSolution import *

DIRNAME = os.path.dirname(__file__)
FILENAME: str = os.path.join(DIRNAME, 'test_instances/small/inst_50_4_00001')
FILENAME_COMPET: str = os.path.join(DIRNAME, 'competition_instances/inst_50_4_00001')
FILENAME_COMPET_2: str = os.path.join(DIRNAME, 'competition_instances/inst_200_20_00001')
# FILENAME_LARGE: str = os.path.join(DIRNAME, 'test_instances/la')
if __name__ == '__main__':
    parser = get_settings_parser()
    parser.set_defaults(mh_titer=10) # number of iterations
    parser.set_defaults(mh_ttime=10) # time limit

    ###INIT
    mWCCPInstance = v2_MWCCPInstance(FILENAME_COMPET)  # FILENAME
    mWCCPSolution = v2_MWCCPSolution(mWCCPInstance)

    ###Parser arguments
    parser.add_argument("--alg", type=str, default='grasp', help='optimization algorithm to be used '
                                                                '(const_det, const_rand, ls, vnd, grasp, gvns, sa, ts)')
    parser.add_argument("--inst_file", type=str, default=mWCCPInstance,
                        help='problem instance file')
    parser.add_argument("--meths_ch", type=int, default=1,
                        help='number of construction heuristics to be used')
    parser.add_argument("--meths_li", type=int, default=1,
                        help='number of local improvement methods to be used')
    parser.add_argument("--meths_sh", type=int, default=5,
                        help='number of shaking methods to be used')
    
    #change manually for step function
    parser.add_argument("--meths_ls_step", type=str, default='first',
                        help='which step function should be used for local search'
                        '(first, best, random)')
    #change manually for move function
    parser.add_argument("--meths_ls_move", type=str, default='swap',
                        help='which step function should be used for local search'
                        '(swap, shift)')
    
    parse_settings(None, 0)

    init_logger()
    logger = logging.getLogger("pymhlib")
    logger.info("pymhlib demo for solving %s", "MWCCP")
    logger.info(get_settings_as_str())
    logger.info("%s instance read:\n%s", "MWCCP", str(mWCCPInstance))

    #util.text = "Initial probleminstance"
    #util.draw_instance(u=mWCCPSolution.instance_u, x=mWCCPSolution.x, w=mWCCPSolution.instance_w)
    
    #
    # alg = GVNS(mWCCPSolution,
    #            [Method(f"construct{i}", MWCCPSolution.construct, i) for i in range(settings.meths_ch)],
    #            [Method(f"local-2opt{i}", MWCCPSolution.local_improve, i) for i in range(1, settings.meths_li + 1)],
    #            [Method(f"sh{i}", MWCCPSolution.shaking, i) for i in range(1, settings.meths_sh + 1)],
    #            None, False)
    #

    if settings.alg == 'const_det':
        alg = GVNS(mWCCPSolution,
                    [Method(f"construct{i}", v2_MWCCPSolution.construct, i) for i in range(settings.meths_ch)],
                    [],
                    [],
                    None, False)
    elif settings.alg == 'const_rand':
        alg = GVNS(mWCCPSolution,
                    [Method(f"construct{i}", v2_MWCCPSolution.construct_random, i) for i in range(settings.meths_ch)],
                    [],
                    [],
                    None, False)
    elif settings.alg == 'ls':
        if settings.meths_ls_move == 'swap':
            if settings.meths_ls_step == 'first':
                alg = GVNS(mWCCPSolution,
                    [Method(f"construct{i}", v2_MWCCPSolution.construct, i) for i in range(settings.meths_ch)],
                    [Method(f"local-2opt{i}", v2_MWCCPSolution.ls_two_swap_first, i) for i in range(1, settings.meths_li + 1)],
                    [],
                    None, False)
            elif settings.meths_ls_step == 'best':
                alg = GVNS(mWCCPSolution,
                    [Method(f"construct{i}", v2_MWCCPSolution.construct, i) for i in range(settings.meths_ch)],
                    [Method(f"local-2opt{i}", v2_MWCCPSolution.ls_two_swap_best, i) for i in range(1, settings.meths_li + 1)],
                    [],
                    None, False)
            elif settings.meths_ls_step == 'random':
                alg = GVNS(mWCCPSolution,
                    [Method(f"construct{i}", v2_MWCCPSolution.construct, i) for i in range(settings.meths_ch)],
                    [Method(f"local-2opt{i}", v2_MWCCPSolution.ls_two_swap_random, i) for i in range(1, settings.meths_li + 1)],
                    [],
                    None, False)   
            else: 
                pass
        elif settings.meths_ls_move == 'shift':
            pass
    elif settings.alg == 'vnd':
        pass
    elif settings.alg == 'grasp':
        alg = GVNS(mWCCPSolution,
                    [Method(f"construct{i}", v2_MWCCPSolution.construct_grasp, i) for i in range(settings.meths_ch)],
                    [Method(f"local-2opt{i}", v2_MWCCPSolution.grasp, i) for i in range(1, settings.meths_li + 1)],
                    [],
                    None, False) 
    elif settings.alg == 'gvns':
        pass
    elif settings.alg == 'sa':
        pass
    elif settings.alg == 'ts':
        pass

    alg.run()
    logger.info("")
    alg.method_statistics()
    alg.main_results()
    # util.text = "Applied heuristic optimization methods"
    # util.draw_instance(u=mWCCPSolution.instance_u, x=mWCCPSolution.x, w=mWCCPSolution.instance_w)


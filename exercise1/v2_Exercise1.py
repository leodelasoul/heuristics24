import logging
import os

from pymhlib.gvns import GVNS
from pymhlib.log import init_logger
from pymhlib.scheduler import Method
from pymhlib.settings import parse_settings, settings, get_settings_parser, get_settings_as_str

from v2_MWCCPInstance import *
from v2_MWCCPSolution import *

DIRNAME = os.path.dirname(__file__)
FILENAME: str = os.path.join(DIRNAME, '../test_instances/small/inst_50_4_00001')
FILENAME_COMPET_1: str = os.path.join(DIRNAME, '../competition_instances/inst_50_4_00001')
FILENAME_COMPET_2: str = os.path.join(DIRNAME, '../competition_instances/inst_200_20_00001')
FILENAME_COMPET_3: str = os.path.join(DIRNAME, '../competition_instances/inst_500_40_00003')
FILENAME_COMPET_4: str = os.path.join(DIRNAME, '../competition_instances/inst_500_40_00012')
FILENAME_COMPET_5: str = os.path.join(DIRNAME, '../competition_instances/inst_500_40_00021')

FILENAME_TEST_SMALL: str = os.path.join(DIRNAME, '../test_instances/small/inst_50_4_00003')
FILENAME_TEST_MEDIUM: str = os.path.join(DIRNAME, 'test_instances/medium/inst_50_4_000010')
FILENAME_TEST_MEDIUM_LARGE: str = os.path.join(DIRNAME, 'test_instances/medium_large/inst_500_40_00003')
FILENAME_TEST_LARGE: str = os.path.join(DIRNAME, '../test_instances/large/inst_1000_60_00004')

FILENAME_TUNING_SMALL: str = os.path.join(DIRNAME, '../competition_instances/inst_500_40_00003')
FILENAME_TUNING_MEDIUM: str = os.path.join(DIRNAME, '../competition_instances/inst_500_40_00003')
FILENAME_TUNING_MEDIUM_LARGE: str = os.path.join(DIRNAME, '../competition_instances/inst_500_40_00003')
FILENAME_TUNING_LARGE: str = os.path.join(DIRNAME, '../competition_instances/inst_500_40_00003')
# FILENAME_LARGE: str = os.path.join(DIRNAME, 'test_instances/la')
if __name__ == '__main__':
    parser = get_settings_parser()
    parser.set_defaults(mh_titer=1500) # number of iterations
    parser.set_defaults(mh_ttime=300) # time limit

    ###INIT
    mWCCPInstance = v2_MWCCPInstance(FILENAME_COMPET_1)  # FILENAME
    mWCCPSolution = v2_MWCCPSolution(mWCCPInstance)

    ###Parser arguments
    parser.add_argument("--alg", type=str, default='const_det', help='optimization algorithm to be used '
                                                                '(const_det, const_rand, ls, vnd, grasp, gvns, sa, ts)')
    parser.add_argument("--inst_file", type=str, default=mWCCPInstance,
                        help='problem instance file')
    parser.add_argument("--meths_ch", type=int, default=5,
                        help='number of construction heuristics to be used')
    parser.add_argument("--meths_li", type=int, default=1,
                        help='number of local improvement methods to be used')
    parser.add_argument("--meths_sh", type=int, default=1,
                        help='number of shaking methods to be used')
    
    #change manually for step function
    parser.add_argument("--meths_ls_step", type=str, default='random',
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
            if settings.meths_ls_step == 'first':
                alg = GVNS(mWCCPSolution,
                    [Method(f"construct{i}", v2_MWCCPSolution.construct, i) for i in range(settings.meths_ch)],
                    [Method(f"local-2opt{i}", v2_MWCCPSolution.ls_shift_first, i) for i in range(1, settings.meths_li + 1)],
                    [],
                    None, False)
            elif settings.meths_ls_step == 'best':
                alg = GVNS(mWCCPSolution,
                    [Method(f"construct{i}", v2_MWCCPSolution.construct, i) for i in range(settings.meths_ch)],
                    [Method(f"local-2opt{i}", v2_MWCCPSolution.ls_shift_best, i) for i in range(1, settings.meths_li + 1)],
                    [],
                    None, False)
            elif settings.meths_ls_step == 'random':
                alg = GVNS(mWCCPSolution,
                    [Method(f"construct{i}", v2_MWCCPSolution.construct, i) for i in range(settings.meths_ch)],
                    [Method(f"local-2opt{i}", v2_MWCCPSolution.ls_shift_random, i) for i in range(1, settings.meths_li + 1)],
                    [],
                    None, False)
            else:
                pass
    elif settings.alg == 'vnd':
        alg = GVNS(mWCCPSolution,
                    [Method(f"construct{i}", v2_MWCCPSolution.construct, i) for i in range(settings.meths_ch)],
                    [Method(f"local-2opt", v2_MWCCPSolution.ls_two_swap_best, 1),
                     Method(f"local-1opt", v2_MWCCPSolution.ls_shift_best, 2)],
                    [],
                    None, False)
    elif settings.alg == 'grasp':
        alg = GVNS(mWCCPSolution,
                    [Method(f"construct{i}", v2_MWCCPSolution.construct_grasp, i) for i in range(settings.meths_ch)],
                    [Method(f"grasp{i}", v2_MWCCPSolution.grasp, i) for i in range(1, settings.meths_li + 1)],
                    [],
                    None, False) 
    elif settings.alg == 'gvns':
        alg = GVNS(mWCCPSolution,
                    [Method(f"construct{i}", v2_MWCCPSolution.construct, i) for i in range(settings.meths_ch)],
                    [Method(f"local-2opt", v2_MWCCPSolution.ls_two_swap_best, 1),
                     Method(f"local-1opt", v2_MWCCPSolution.ls_shift_best, 2)],
                    [Method(f"shaking-2opt", v2_MWCCPSolution.shaking_swap, 1),
                     Method(f"shaking-1opt", v2_MWCCPSolution.shaking_shift, 2)],
                    None, False) 
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


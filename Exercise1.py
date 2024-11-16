import os

from pymhlib.settings import get_settings_parser

import MWCCPInstance
import MWCCPSolution

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


    # #run_optimization('MWCCP', MWCCPInstance, MWCCPSolution, "instance")



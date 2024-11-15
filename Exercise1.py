import MWCCPInstance
import MWCCPSolution
from pymhlib.demos.common import run_optimization, data_dir
from pymhlib.settings import get_settings_parser
import os

DIRNAME = os.path.dirname(__file__)
FILENAME : str = os.path.join(DIRNAME, 'test_instances/small/inst_50_4_00001')

if __name__ == '__main__':
    parser = get_settings_parser()
    parser.set_defaults(mh_titer=1000)
    ### init problem instance
    mWCCPInstance = MWCCPInstance.MWCCPInstance()
    mWCCPInstance.__int__(FILENAME) # FILENAME
    #mWCCPInstance.draw_instance()
    mWCCPInstance.set_problem_instance()
    mWCCPSolution = MWCCPSolution.MWCCPSolution(mWCCPInstance)
    mWCCPInstance.get_instance()

    # #run_optimization('MWCCP', MWCCPInstance, MWCCPSolution, "instance")





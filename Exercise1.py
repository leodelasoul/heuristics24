import MWCCPInstance
import MWCCPSolution

if __name__ == '__main__':
    from pymhlib.demos.common import run_optimization, data_dir
    from pymhlib.settings import get_settings_parser
    parser = get_settings_parser()
    parser.set_defaults(mh_titer=1000)

    lol = MWCCPInstance.MWCCPInstace
    lol.__int__("instance")
    print(lol.get_instance())
    #run_optimization('MWCCP', MWCCPInstance, MWCCPSolution, "instance")
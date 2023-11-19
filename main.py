import sys
from tqdm import tqdm_notebook

import Utils
import Train

def main():
    for arg in sys.argv:
        print(arg)

    if(len(sys.argv) == 15):
        opt = Utils.arg_parser_subst(sys.argv)
    else:
        opt = Utils.arg_parser_subst()

    Train.train(opt)

    return


if __name__ == "__main__":
    main()

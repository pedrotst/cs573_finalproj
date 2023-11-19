import sys
from tqdm import tqdm_notebook

import Utils
import Train

def main():
    for arg in sys.argv:
        print(arg)

    opt = Utils.arg_parser_subst(sys.argv)

    Train.train(opt)

    return


if __name__ == "__main__":
    main()

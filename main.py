import sys
from tqdm import tqdm_notebook

import Utils
import Train

def main():
    opt = Utils.arg_parser_subst(sys.argv)
    Train.train(opt)


if __name__ == "__main__":
    main()

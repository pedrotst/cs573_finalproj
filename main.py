import sys
from tqdm import tqdm_notebook

import utils
import train

def main():
    opt = utils.arg_parser_subst(sys.argv)
    train.train(opt)


if __name__ == "__main__":
    main()

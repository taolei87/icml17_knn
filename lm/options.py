
import sys
import argparse

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train",
            type = str,
            default = ""
        )
    argparser.add_argument("--dev",
            type = str,
            default = ""
        )
    argparser.add_argument("--test",
            type = str,
            default = ""
        )
    argparser.add_argument("--hidden_dim", "-d",
            type = int,
            default = 200
        )
    argparser.add_argument("--learning",
            type = str,
            default = "sgd"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = 0.01
        )
    argparser.add_argument("--activation", "-act",
            type = str,
            default = "tanh"
        )
    argparser.add_argument("--batch_size",
            type = int,
            default = 32
        )
    argparser.add_argument("--dropout",
            type = float,
            default = 0.5
        )
    argparser.add_argument("--max_epoch",
            type = int,
            default = 50
        )
    argparser.add_argument("--lr_decay",
            type = float,
            default = 0.9
        )
    argparser.add_argument("--lr_decay_epoch",
            type = int,
            default = 10
        )
    argparser.add_argument("--unroll_size",
            type = int,
            default = 35
        )
    argparser.add_argument("--depth",
            type = int,
            default = 1
        )
    argparser.add_argument("--eps",
            type = float,
            default = 1e-6
        )
    argparser.add_argument("--highway",
            type = int,
            default = 1
        )
    args = argparser.parse_args()
    args = vars(args)
    return args


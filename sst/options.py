import sys
import argparse

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train",
            type = str,
            default = "",
            help = "path to training data"
        )
    argparser.add_argument("--dev",
            type = str,
            default = "",
            help = "path to development data"
        )
    argparser.add_argument("--test",
            type = str,
            default = "",
            help = "path to test data"
        )
    argparser.add_argument("--hidden_dim", "-d",
            type = int,
            default = 200,
            help = "hidden dimensions"
        )
    argparser.add_argument("--learning",
            type = str,
            default = "adam",
            help = "learning method (sgd, adagrad, adam, ...)"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = "0.001",
            help = "learning rate"
        )
    argparser.add_argument("--max_epochs",
            type = int,
            default = 200,
            help = "maximum # of epochs"
        )
    argparser.add_argument("--dropout",
            type = float,
            default = 0.3,
            help = "dropout probability"
        )
    argparser.add_argument("--rnn_dropout",
            type = float,
            default = 0.0,
            help = "dropout probability"
        )
    argparser.add_argument("--lr_decay",
            type = float,
            default = 0.0
        )
    argparser.add_argument("--l2_reg",
            type = float,
            default = 1e-6
        )
    argparser.add_argument("--embedding",
            type = str,
            default = ""
        )
    argparser.add_argument("--batch_size",
            type = int,
            default = 32,
            help = "mini-batch size"
        )
    argparser.add_argument("--depth",
            type = int,
            default = 3,
            help = "number of feature extraction layers (min:1)"
        )
    argparser.add_argument("--activation",
            type = str,
            default = "relu",
            help = "activation function (none, relu, tanh, etc.)"
        )
    argparser.add_argument("--save",
            type = str,
            default = "",
            help = "save model to this file"
        )
    argparser.add_argument("--load",
            type = str,
            default = "",
            help = "load model from this file"
        )
    argparser.add_argument("--pooling",
            type = int,
            default = 1,
            help = "whether to use mean pooling or take the last vector"
        )
    argparser.add_argument("--highway",
            type = int,
            default = 0
        )
    argparser.add_argument("--multiplicative",
            type = int,
            default = 1
        )
    args = argparser.parse_args()
    return args


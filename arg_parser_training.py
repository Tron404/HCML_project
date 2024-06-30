import argparse

class argParser:
    def __init__(self) -> None:
        self.arg_parser = argparse.ArgumentParser()

        # BERT instantiation parameters
        self.arg_parser.add_argument("--loss_func", type=str)
        self.arg_parser.add_argument("--agg", type=str)
        self.arg_parser.add_argument("--freeze", type=str)
        self.arg_parser.add_argument("--regress_size", type=int)
        self.arg_parser.add_argument("--dense", nargs="+", type=lambda x: tuple([int(item) for item in x.split(",")]))
        self.arg_parser.add_argument("--hidden", nargs="+", type=lambda x: int(x))
    
        # training arguments
        self.arg_parser.add_argument("--datasize", type=int)
        self.arg_parser.add_argument("--batches", type=int)
        self.arg_parser.add_argument("--epochs", type=int)
        self.arg_parser.add_argument("--w_decay", type=float)
        self.arg_parser.add_argument("--lr", type=float)
        self.arg_parser.add_argument("--optimizer", type=str)
        self.arg_parser.add_argument("--gpu", type=bool)
        self.arg_parser.add_argument("--num_trials", type=int)
        self.arg_parser.add_argument("--study_name", type=str)
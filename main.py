import torch
import argparse
import os
import numpy as np
from logger_utils import set_log
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from option_critic import train_four_rooms

def main(args):
    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Set logs
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))
    log = set_log(args)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    iterations = 10
    noptions = 30

    # Start train
    for noption in range(1,noptions+1):
        for i in range(iterations):
            train_four_rooms(args, log, tb_writer, noptions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--tau", default=0.01, type=float, 
        help="Target network update rate")
    parser.add_argument(
        "--batch-size", default=50, type=int, 
        help="Batch size for both actor and critic")
    parser.add_argument(
        "--policy-freq", default=2, type=int,
        help="Frequency of delayed policy updates")
    parser.add_argument(
        "--actor-lr", default=0.0001, type=float,
        help="Learning rate for actor")
    parser.add_argument(
        "--critic-lr", default=0.001, type=float,
        help="Learning rate for critic")
    parser.add_argument(
        "--n-hidden", default=200, type=int,
        help="Number of hidden units")
    parser.add_argument(
        "--discount", default=0.99, type=float, 
        help="Discount factor")

    # Misc
    parser.add_argument(
        "--prefix", default="", type=str,
        help="Prefix for tb_writer and logging")
    parser.add_argument(
        "--seed", default=0, type=int, 
        help="Sets Gym, PyTorch and Numpy seeds")

    args = parser.parse_args()

    # Set log name
    args.log_name = "exp1"

    main(args=args)




import argparse
import os
from util import util
import torch
import models
import data

class TrainOptions():
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        

    def initialize(self, parser):
 
        # network parameters
        parser.add_argument('--result', type=str, default='result',help='The number of folder for the save of model and loss')
        parser.add_argument('--gpu', type=int, default=0)
        parser.add_argument('--batch', type=int, default=256)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--gamma', type=float, default=0.1)
        parser.add_argument('--schedular', type=int, default=5)
        # dataset parameters 
        parser.add_argument('--RF1', type=str, default='_sampled_RF1')
        parser.add_argument('--testset', type=str, default='_random_RF')
        # LSTM structure parameters 
        parser.add_argument('--hidden_size', type=int, default=512,help='Hidden size of LSTM')
        parser.add_argument('--layer', type=int, default=3,help='Number of layers for each LSTM')
        parser.add_argument('--input', type=int, default=5,help='size of the input vector: MTC-MRF signal + four scan parameters')
        parser.add_argument('--out_dim', type=int, default=4,help='dimension of output: number of target tissue parameters')
        parser.add_argument('--Bi', type=str, default='bi',help='Either bi-directional or uni-directional')
        parser.add_argument('--dropout', type=float, default=0)
        
        self.isTrain = True
        return parser
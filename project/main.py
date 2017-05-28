import tensorflow as tf
import utils
import resnet_model

from resnet import *
from flags import *


def main():
    resnet = Resnet()
    resnet.train()
        
        
if __name__ == "__main__":
    main()

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.cluster import KMeans
from utils.utils import *

TRAIN_PATH = 'data/train'
TEST_PATH = 'data/test'
VAL_PATH = 'data/valid'
MODEL_PATH = 'models/model.h5'
GRID_SIZE = 7
ANCHORS = 1
NUM_CLASSES = 1
TRAIN_VGG = False
LEARNING_RATE = 1e-4

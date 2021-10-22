'''
Predicting Modalities in a single cell using the transformer: a deep learning architecture.
'''
__author__ = 'Vedu Mallela: vedu.mallela@gmail.com, Simon Lee: siaulee@ucsc.edu'

import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import logging

from transformers import BertGenerationTokenizer, BertGenerationEncoder, BertGenerationDecoder, BertGenerationConfig
import torch

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error




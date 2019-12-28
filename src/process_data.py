import os
import json
import numpy as np
import random
import networkx as nx
from data_load import Data_loading

class Data_process(Data_loading):
    """
    Data process, cut edges, nodes, create transductive and inductive training ,testing set
    """
    def __init__(self,data_set):
        Data_loading.__init__(self,data_set)

    def cut_edges(self):




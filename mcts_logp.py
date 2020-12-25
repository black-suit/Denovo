import numpy as np
import time
import math
import random as pr

from load_model import loaded_model
from add_node_type import chem_kn_simulation, expanded_node, node_to_add, predict_smile, make_input_smile, check_node_type
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket

class chemical:

class Node:

def MCTS(root,verbose = False):

def UCTchemical():
  one_search_start_time = time.time()
  time_out = one_search_start_time + 60*10
  state = chemical()
  best = MCTS(root=state,verbose=False)
  
  return best

if __name__ == "__main__":
  smile_old = zinc_data_with_bracket_original()
  val,smile = zinc_processed_with_bracket(smile_old)
  
  logP_values = np.loadtxt('logP_values.txt')
  SA_scores = np.loadtxt('SA_scores.txt')
  #cycle_scores = np.loadtxt('cycle_scores.txt')
  SA_mean = np.mean(SA_scores)
  SA_std = np.std(SA_scores)
  logP_mean = np.mean(logP_values)
  #cycle_mean = np.mean(cycle_scores)
  #cycle_std = np.std(cycle_scores)
  model = loaded_model()
  valid_compound = UCTChemical()

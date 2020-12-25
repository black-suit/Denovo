from math import *
import numpy as np
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import RDConfig
import os
#import sys
#sys.path.append(os.path.join(RDConfig.RDContribDir,'SA_Score'))
#import sascorer
from rdkit.Chem import MolFromSmiles, MolToSmiles
import math
from rdkit.Chem import AllChem, rdMolDescriptors

def expanded_node(model,state,val):
  all_nodes = []
  end = "\n"
  
  position = []
  position.extend(state)
  total_generated = []
  new_compound = []
  get_int_old = []
  for j in range(len(position)):
    get_int_old.append(val.index(position[j]))
  get_int = get_int_old
  
  x = np.reshape(get_int,(1,len(get_int)))
  x_pad = tf.keras.preprocessing.sequence.pad_sequences(x,maxlen=82,dtype='32',padding='post',truncating='pre',value=0.)
  
  for i in range(30):
    predictions = model.predict(x_pad)
    preds = np.log(preds) / 1.0
    preds = np.exp(preds) / np.sum(np.exp(preds))
    next_probas = np.random.multinomial(1,preds,1)
    next_int = np.argmax(next_probas)
    all_nodes.append(next_int)
  all_nodes = list(set(all_nodes))
  print(all_nodes)
  
  return all_nodes

def node_to_add(all_nodes,val):
  return added_nodes

def chem_kn_simulation(model,state,val,added_nodes):
  return all_posible

def predict_smile(all_posible,val):
  return new_compound

def make_input_smile(generate_smile):
  return new_compound

def check_node_type(new_compound,SA_mean,SA_std,logP_mean,logP_std,cycle_mean,cycle_std):
  return node_index,score,valid_compound,all_smile

def logp_calculation(new_compound):
  print(new_compound[0])
  logp_value = []
  all_smile = []
  distance = []]
  m = Chem.MolFromSmiles(str(new_compound[0]))
  try:
    if m is not None:
      logp = Descriptors.MolLogP(m)
      valid_smile.append(new_compound)
    else:
      logp = -100
  except:
    logp = -100
  all_smile.append(str(new_compound[0]))
  
  return logp,valid_smile,all_smile
  return logp,valid_smile,all_smile

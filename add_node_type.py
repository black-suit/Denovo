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

from expanded_node(model,state,val):
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

from rdkit import Chem
from rdkit.Chem import AllChem

def generateconf(m,n=20)
  ps = AllChem.ETKDGv2()
  ps.pruneRmsThresh = 0.2
  ps.numThreads = 0
  mh = Chem.AddHs(m)
 
  cids = AllChem.EmbedMultipleConfs(mh,n,maxAttempts=500,enforceChirality=False,ps)
  es = []

  for cid in cids:
    AllChem.UFFOptimizeMolecule(mh,maxIters=1000,confId=cid)
    ff = AllChem.UFFGetMoleculeForceField(mh,confId=cid)
    ff.Minimize()
    e = ff.CalcEnergy
    es.append((e,cid))
  
  es.sort()
  ind_lowest=es[0][1]
  return mh,ind_lowest

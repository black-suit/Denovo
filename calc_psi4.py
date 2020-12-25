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

def molblock(mh,ind_lowest):
  mb = Chem.MolToMolBlock(mh,ind_lowest)
  i,j,k = 96,68,95
  n = mh.GetNumAtoms() - 1
  mb2 = mb[i]+' '+b[j:k]
  for _ in range(n):
    i+=70
    j+=70
    k+=70
    mb2 += "\n"+mb[i]+' '+mb[j:k]
  return mb2

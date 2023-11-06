import pyvista as pv
import numpy as np
import json
import os
import os.path as osp

from aorta_lib.hemodyn import data

def read_shape_coeff_from_dict(dict, n_coeffs):
  modes = np.array(list(dict.keys()), dtype=int)
  val = np.array(list(dict.values()), dtype=float)
  coeffs = np.zeros(n_coeffs)
  coeffs[modes] = val
  return coeffs

array_names = [
    #'ecaps',
    'tawss',
    'osi',
    #'atawss'
    'wss_sis',
    #'tap',
    #'p_sis'
]

model_path = 'odir_noLATIN1/hemo_interpolators_gaus_reg_pca_new.npy'
dataRoot = '../../examples/alignedDatasetBiomarkers_new/'
dataset_path = "datasetROM_new_noLATIN1.json"

# open post-processed simulation dataset
datadict = data.read_datalist(dataset_path, folds=range(5),
                              dataRoot=dataRoot, keys_to_edit=["model"])


models = np.load(model_path, allow_pickle=True)[()]
interpDict = models['reg']
pcaDict = models['pca']

odir = "odir_noLATIN1_evaluate"
if not osp.isdir(odir):
  os.makedirs(odir)

for sample in datadict:
  fold = sample['fold']
  w = read_shape_coeff_from_dict(sample["coeffs"], 47)
  m = pv.read(sample["model"])
  print(sample["name"], end=" ")
  for array_name in array_names:
    print(array_name, end=" ")
    interp = interpDict[array_name]
    array_transformed, std_transformed = interp.predict(w[None,:25], return_std=True)
    array = np.squeeze(pcaDict[array_name].inverse_transform(array_transformed))
    std = np.abs(np.squeeze(np.dot(std_transformed, pcaDict[array_name].components_)))
    m[array_name+"_ROM"] = array

    interp = interpDict[array_name+f"_fold{fold}"]
    array_transformed_val, std_transformed_val = interp.predict(w[None,:25], return_std=True)
    array_val = np.squeeze(pcaDict[array_name+f"_fold{fold}"].inverse_transform(array_transformed_val))

    m[array_name+"_ROM_val"] = array_val

  m.save(osp.join(odir, sample["name"]+".vtp"))
  print()

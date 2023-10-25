import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

import json
import os
import os.path as osp
import sys
import tqdm

import sklearn
import sklearn.gaussian_process
import sklearn.decomposition
import sklearn.neighbors as nb

from aorta_lib.ssm import generateShape as gs
from aorta_lib.hemodyn import data

SSM, _ = gs.load_SSM("../ssm/SSM_47shapes_relaxed_corrected.npy")
ssm_std = np.sqrt(SSM['model'].explained_variance_)
k = 5 # k nearest neigbours

def train_ROM(hemo_dim, X, Y, array_name, plot=False):
  reg = sklearn.gaussian_process.GaussianProcessRegressor(
    kernel=None,
    optimizer='fmin_l_bfgs_b',
    n_restarts_optimizer=0,
    normalize_y=False,
    random_state=1,
  )
  pca = sklearn.decomposition.PCA(n_components=hemo_dim, whiten=False)
  Y_PCA = pca.fit_transform(Y)
  print(f"    X shape: {X.shape}")
  print(f"    Y_PCA shape: {Y_PCA.shape}")
  reg.fit(X, Y_PCA)
  print('    regressor fitted')
  return reg, pca


def build_knn(X, Y, distance_callable, k=1):
  # build a knn tree of samples in X to efficiently find k-nearest neigbours of each sample in Y
  # returns distances and indices of neigbours
  neigs = nb.NearestNeighbors(n_neighbors=k, algorithm='auto', metric=distance_callable)
  neigs = neigs.fit(X)
  return neigs.kneighbors(Y, k, return_distance=True)


def evaluate_ROM(reg, pca, X, Y, X_test, Y_test, array_name, model_odir=None):
  Y_pca = pca.transform(Y)
  Y_pca_pred = reg.predict(X)
  Y_rom = pca.inverse_transform(Y_pca_pred)
  u = ( (Y_rom - Y)**2 ).sum()
  v = ( (Y - Y.mean())**2 ).sum()
  score_pca_trn = reg.score(X, Y_pca)
  score_recon_trn = 1-u/v
  rmse_trn = np.sqrt(((Y_rom - Y)**2).mean())
  abse_trn = (np.abs(Y_rom - Y)).mean()

  if X_test is None:
    score_pca_test = 0.
    score_recon_test = 0.
    rmse_test = 0.
    abse_test = 0.
  else:
    Y_pca = pca.transform(Y_test)
    Y_pca_pred = reg.predict(X_test)
    Y_rom = pca.inverse_transform(Y_pca_pred)
    u = ( (Y_rom - Y_test)**2 ).sum()
    v = ( (Y_test - Y_test.mean())**2 ).sum()
    score_pca_test = reg.score(X_test, Y_pca)
    score_recon_test = 1-u/v
    rmse_test = np.sqrt(((Y_rom - Y_test)**2).mean())
    abse_test = (np.abs(Y_rom - Y_test)).mean()
    if model_odir is not None:
      dist, ind = build_knn(X_trn, X_test, distance_shapes, k=k)
      dist = dist.mean(-1)
      plt.title(f"scatter plot for {array_name}, fold {fold}")
      plt.xlabel("dist from nn in training")
      plt.ylabel(f"abs error for {array_name}")
      plt.scatter(dist, np.abs(Y_rom - Y_test).mean(axis=1), s=0.6)
      plt.savefig(osp.join(model_odir,f"dist_vs_abse_{array_name}_fold_{fold}_k_{k}.png"), dpi=150)
      plt.close()
      np.savetxt(osp.join(model_odir,f"dist_vs_abse_{array_name}_fold_{fold}_k_{k}.txt"), np.c_[dist,np.abs(Y_rom - Y_test).mean(axis=1)])
    #if plot:
    #  dist, ind = build_knn(X_trn, X_test, distance_shapes, k=k)
    #  dist = dist.mean(-1)
    #  axs[fold].scatter(dist, np.abs(Y_rom - Y_test).mean(axis=1), s=0.6)


  print("    " + array_name)
  print("    {:25s} {:.4f} | {:.4f}".format("score           PCA:", score_pca_trn, score_pca_test))
  print("    {:25s} {:.4f} | {:.4f}".format("score Reconstructed:", score_recon_trn, score_recon_test))
  print("    {:25s} {:.4f} | {:.4f}".format("RMSE  Reconstructed:",        rmse_trn,        rmse_test))
  print("    {:25s} {:.4f} | {:.4f}".format("ABSE  Reconstructed:",        abse_trn,        abse_test))


def __create_train_test_sets__(dataset, fold, nfolds):
  # creates only the shape coefficients for statistical analysis
  if fold is None:
    X = np.array([s["shape_coeff"] for s in dataset])
    return X, None

  folds = list(range(nfolds))
  train_folds = [f for f in folds if f != fold]
  test_folds = [f for f in folds if f == fold]
  X_trn = np.array([s["shape_coeff"] for s in dataset if s["fold"] in train_folds])
  X_test = np.array([s["shape_coeff"] for s in dataset if s["fold"] in test_folds])
  return X_trn, X_test


def create_train_test_sets(dataset, fold, nfolds):
  if fold is None:
    X = np.array([s["shape_coeff"] for s in dataset])
    Y = np.array([s[array_name] for s in dataset])
    return X, Y, None, None

  folds = list(range(nfolds))
  train_folds = [f for f in folds if f != fold]
  test_folds = [f for f in folds if f == fold]
  X_trn = np.array([s["shape_coeff"] for s in dataset if s["fold"] in train_folds])
  Y_trn = np.array([s[array_name] for s in dataset if s["fold"] in train_folds])
  X_test = np.array([s["shape_coeff"] for s in dataset if s["fold"] in test_folds])
  Y_test = np.array([s[array_name] for s in dataset if s["fold"] in test_folds])
  return X_trn, Y_trn, X_test, Y_test

def read_shape_coeff_from_dict(dict, n_coeffs):
  modes = np.array(list(dict.keys()), dtype=int)[:n_coeffs]
  val = np.array(list(dict.values()), dtype=float)[:n_coeffs]
  coeffs = np.zeros(n_coeffs)
  coeffs[modes] = val
  return coeffs

def distance_shapes(x, y):
  # weighted euclidean distance based on ssm_std
  return np.sqrt((((x-y)*ssm_std[:ssm_pca_ncomps])**2).sum(axis=-1))


if __name__ == "__main__":

  hemo_pca_ncomps = 250
  ssm_tot_pca_ncomps = 47
  ssm_pca_ncomps = 25
  array_names = ['tawss', 'ecap', 'osi', 'tap', 'p_sis', 'wss_sis']
  dataRoot = '../../examples/alignedDatasetBiomarkers_new/'
  dataset_path = "datasetROM_new.json"
  model_odir = "odir"
  if not osp.isdir(model_odir):
    os.makedirs(model_odir)
  else:
    print('odir already exists. exiting')
    exit

  # read dataset
  datadict = data.read_datalist(dataset_path, folds=range(5), dataRoot=dataRoot, keys_to_edit=["model"])
  shape_coeffs = []
  dataset = []
  for i in tqdm.tqdm(
          range(len(datadict)),
          desc='Loading simulations', ncols=100):
    sample = datadict[i]
    data_dict = {}
    data_dict["name"] = sample["name"]
    data_dict["fold"] = sample["fold"]
    data_dict["shape_coeff"] = read_shape_coeff_from_dict(sample["coeffs"], ssm_tot_pca_ncomps)[:ssm_pca_ncomps]
    m = pv.read(sample["model"])
    for array_name in array_names:
      data_dict[array_name] = (m[array_name] if m[array_name].ndim == 1 else np.sqrt((m[array_name]**2).sum(-1))).astype(np.float32)
    dataset.append(data_dict)

  interpolatorDict = {}
  pcaDict = {}

  for array_name in ['tawss', 'osi', 'wss_sis',]:# 'tap', 'p_sis']:
    print('array name: ', array_name)
    nfolds = 5 #TODO read nfolds from datasetROM.json

    regs = []
    pcas = []
    for fold in range(nfolds):
      print("  fold: ", fold)
      X_trn, Y_trn, X_test, Y_test = create_train_test_sets(dataset, fold, nfolds)
      reg, pca = train_ROM(hemo_pca_ncomps, X_trn, Y_trn, array_name)
      evaluate_ROM(reg, pca, X_trn, Y_trn, X_test, Y_test, array_name, model_odir=model_odir)
      regs.append(reg)
      pcas.append(pca)

    print()
    print("==fold: ", "tot")
    X_trn, Y_trn, X_test, Y_test = create_train_test_sets(dataset, None, nfolds)
    reg, pca = train_ROM(hemo_pca_ncomps, X_trn, Y_trn, array_name)
    evaluate_ROM(reg, pca, X_trn, Y_trn, X_test, Y_test, array_name)
    interpolatorDict[array_name] = reg
    pcaDict[array_name] = pca
    for fold in range(nfolds):
      interpolatorDict[array_name+f"_fold{fold}"] = regs[fold]
      pcaDict[array_name+f"_fold{fold}"] = pcas[fold]

  # save models
  totDict = {'reg':interpolatorDict, 'pca':pcaDict}
  np.save(osp.join(model_odir, 'hemo_interpolators_gaus_reg_pca_new.npy'), totDict)





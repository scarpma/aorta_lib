from aorta_lib.ssm.generateShape import *
import json
from random import shuffle
import tqdm

# search for available simulations (only post-processed surface models)
simDir = '../../examples/alignedDatasetBiomarkers_new'
paths = glob.glob(simDir + '/*.vtp')
paths = sorted(paths)

# load ssm to encode shapes into shape-coefficients
SSM_PATH = '../ssm/SSM_47shapes_relaxed_corrected.npy'
SSM, refPvModel = load_SSM(SSM_PATH)

# initialize dataset dict
datadict = dict(
  name = "Aorta",
  description = "Aorta simulations",
  reference = "Fondazione Monasterio, BioCardioLab, Martino Andrea Scarpolini",
  licence = "",
  relase = "1.0 15/12/2022",
  numTraining = 0,
  numTest = 0,
  training = [],
  test = [],
)
datalist = datadict["training"]

# populate dataset
for ii, path in tqdm.tqdm(enumerate(paths)):
  tmp = {}
  filename = osp.split(path)[-1]
  tmp["model"] = filename
  name = osp.splitext(filename)[0]
  tmp["name"] = name

  m = pv.read(path)
  shape_coeff = np.around(reduceShapeCore(SSM=SSM, pvModel=m).astype(np.float32), 4)
  shape_coeff[shape_coeff==0.] = 0.
  if shape_coeff[-1] != 0.:
    shape_coeff[-1] = 0.
  coeffs = {str(i):str(shape_coeff[i]) for i in range(shape_coeff.shape[0]) if shape_coeff[i] != 0.}
  tmp["coeffs"] = coeffs
  datalist.append(tmp)

datadict["numTraining"] = len(datalist)

# create random folds
n_folds = 5
folds = [i%5 for i in range(len(datalist))]
shuffle(folds)
for ii, sample in enumerate(datalist):
  #print("| {:3} | {:14} | {:1} |".format(ii, sample['name'][:14], folds[ii]))
  sample['fold'] = folds[ii]

# save dataset in json
with open("datasetROM_new.json", "w") as f:
  json.dump(datadict, f, indent=2)



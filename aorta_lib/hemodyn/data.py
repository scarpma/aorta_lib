import json
import os.path as osp
import re

def read_datalist(json_path, folds, dataRoot, keys_to_edit=["image","label"]):
    # deals with kfold cross-validation already present in the datalist
    with open(json_path, 'r') as json_file:
      datalist = json.load(json_file)["training"]
    
    new_datalist = [sample for sample in datalist if sample["fold"] in folds]
    for sample in new_datalist:
        for key in keys_to_edit:
            if "./" in sample[key]:
                internal_path = sample[key].replace("./", "") # remove leading './' to join with directory path
            else:
                internal_path = sample[key]
            sample[key] = osp.join(dataRoot, internal_path)
    
    return new_datalist


def get_sample_group(name):
    n_commas = name.count(',')
    n_us= name.count('_')
    if n_commas == 0 and n_us == 0:
      if re.search('[a-zA-Z]', name):
        return 'PS'
      else:
        return 'RG'
    elif n_commas > 3:
      return 'LATIN1'
    elif n_commas <= 3 and n_commas>=1:
      return 'LATIN2'
    elif n_commas == 0 and n_us>0:
      return 'DIAG'
    else:
      return 'UNKNOWN'

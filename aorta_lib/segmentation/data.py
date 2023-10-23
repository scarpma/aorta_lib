import os.path as osp
import json

def load_datalist(datalist_path, splits_, load_keys=None, patients=None):
    # old implementation
    if 'all' in patients:
        patients = None

    with open(datalist_path, 'r') as f:
        datalist = json.load(f)
    print('internal datalist', datalist)

    dataRoot = osp.split(datalist_path)[0]
    splits = ['training', 'test', 'validation']
    splits = [split for split in splits if split in datalist and len(datalist[split])!=0]
    for split in splits:
        keys = ['image', 'label']
        for i in range(len(datalist[split])):
            for key in keys:
                if datalist[split][i][key] is not None and datalist[split][i][key][:2] == './':
                    datalist[split][i][key] = osp.join(dataRoot, datalist[split][i][key][2:])

    datalist_ = []
    for split in splits_:
        datalist_ += datalist[split]

    if patients is not None:
        datalist_ = [datalist_[i] for i in range(len(datalist_)) if datalist_[i]['name'] in patients]
    if load_keys is not None:
        datalist_ = [{key:dict_[key] for key in load_keys} for dict_ in datalist_]

    if len(datalist_) == 0:
        print('WARNING: datalist is empty')

    return datalist_


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


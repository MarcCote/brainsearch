#!/usr/bin/env python
import os
from os.path import join as pjoin
import numpy as np
import json
from collections import OrderedDict, defaultdict


FILE_TYPE = 'fa.nii'
#FILE_TYPE = 't1_brain.nii.gz'
defects = [line.split()[0] for line in open(pjoin("/home/cotm2719/research/data/neuroimaging/PPMI/", "defect.txt"))]
data_dir = "/home/cotm2719/research/data/neuroimaging/PPMI/registered/FA"

filenames = os.listdir(data_dir)

controls = defaultdict(lambda: [])
parkinsons = defaultdict(lambda: [])

for filename in filenames:
    path = pjoin(data_dir, filename)
    name = filename[:-len(".nii.gz")]
    if "_".join(name.split("_")[1:]) in defects:
       continue

    infos = name.split("_")
    subject_type = infos[0]
    subject_id = infos[1]
    date = infos[2]
    session_id1 = infos[3]
    session_id2 = infos[4]

    if subject_type == "control":
        controls[subject_id].append((path, name))
    elif subject_type == "parkinson":
        parkinsons[subject_id].append((path, name))
    else:
        print "Unknown subject type: {}".format(subject_type)

#from itertools import chain
#controls = list(chain(*controls.values()))
#parkinsons = list(chain(*parkinsons.values()))
# Take only the first image per subject
controls = [control[0] for control in controls.values()]
parkinsons = [parkinson[0] for parkinson in parkinsons.values()]


def split(arr, percents):
    percents = np.asarray(percents)
    percents = percents / percents.sum()

    sizes = np.zeros(len(percents)+1, dtype='int')
    for i, p in enumerate(percents, start=1):
        sizes[i] = int(np.ceil(len(arr) * p))

    split_indices = np.cumsum(sizes)
    return [arr[start:end] for start, end in zip(split_indices[:-1], split_indices[1:])]


rng = np.random.RandomState(42)
rng.shuffle(controls)
rng.shuffle(parkinsons)

#percents = (0.7, 0.15, 0.15)
percents = (0.5, 0.5)
controls_splitted = split(controls, percents)
parkinsons_splitted = split(parkinsons, percents)


def save_dict_to_json_file(path, dictionary):
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': ')))


def build_json(infos, groups, name, rng=np.random.RandomState(42)):
    data = OrderedDict()
    data["name"] = name
    data["type"] = "nifti"

    sources = []
    infos_and_groups = zip(infos, groups)
    rng.shuffle(infos_and_groups)
    for i, (info, group) in enumerate(infos_and_groups):
        source = {}
        source["path"] = info[0]
        source["label"] = group
        source["name"] = info[1]
        sources.append(source)

    data["sources"] = sources

    return data

# Trainset
nb_controls = len(controls_splitted[0])
nb_patients = len(parkinsons_splitted[0])
data = controls_splitted[0] + parkinsons_splitted[0]
groups = [0] * nb_controls + [1] * nb_patients

json_trainset = build_json(data, groups, "rPPMI-train", rng)
save_dict_to_json_file("rPPMI_trainset.json", json_trainset)

# Validset
nb_controls = len(controls_splitted[1])
nb_patients = len(parkinsons_splitted[1])
data = controls_splitted[1] + parkinsons_splitted[1]
groups = [0] * nb_controls + [1] * nb_patients
json_validset = build_json(data, groups, "rPPMI-valid", rng)
save_dict_to_json_file("rPPMI_validset.json", json_validset)

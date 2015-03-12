#!/usr/bin/env python
import os
from datetime import datetime
from os.path import join as pjoin
import numpy as np
import json
from collections import OrderedDict


FILE_TYPE = 'fa.nii'
#FILE_TYPE = 't1_brain.nii.gz'
data_dir = "/home/cotm2719/research/data/neuroimaging/PPMI/"

defects = [line.split()[0] for line in open(pjoin(data_dir, "defect.txt"))]

all_infos = np.recfromcsv(pjoin(data_dir, 'infos.csv'))

# Keep only modality=T1
infos = all_infos[[i for i, modality in enumerate(all_infos['modality']) if modality.lower() == "t1"]]

uids = os.listdir(pjoin(data_dir, "images"))

controls_t1 = []
patients_t1 = []
for subject_id, research_group, study_date, image_id in zip(infos['subject_id'], infos['research_group'], infos['study_date'], infos['image_id']):
    formatted_study_date = datetime.strftime(datetime.strptime(study_date, '%m/%d/%Y'), '%Y-%m-%d')
    uid_prefix = "{subject_id}_{study_date}_{image_id}".format(subject_id=subject_id, study_date=formatted_study_date, image_id=image_id)
    possible_uids = [uid for uid in uids if uid.startswith(uid_prefix)]
    if len(possible_uids) > 0:
        uid = possible_uids[0]  # Take the first one as they contain all the same T1.nii

        # If in defect list, skip it
        if uid in defects:
            continue

        t1_path = pjoin(data_dir, 'images', uid, 'work', FILE_TYPE)
        if os.path.isfile(t1_path):
            if research_group.lower() == "pd":
                patients_t1.append(t1_path)
            elif research_group.lower() == "control":
                controls_t1.append(t1_path)


def split(arr, percents):
    percents = np.asarray(percents)
    percents = percents / percents.sum()

    sizes = np.zeros(len(percents)+1, dtype='int')
    for i, p in enumerate(percents, start=1):
        sizes[i] = int(np.ceil(len(arr) * p))

    split_indices = np.cumsum(sizes)
    return [arr[start:end] for start, end in zip(split_indices[:-1], split_indices[1:])]


rng = np.random.RandomState(42)
rng.shuffle(controls_t1)
rng.shuffle(patients_t1)

percents = (0.7, 0.15, 0.15)
controls_splitted = split(controls_t1, percents)
patients_splitted = split(patients_t1, percents)


def save_dict_to_json_file(path, dictionary):
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': ')))


def build_json(paths, groups, name, rng=np.random.RandomState(42)):
    data = OrderedDict()
    data["name"] = name
    data["type"] = "nifti"

    sources = []
    paths_and_groups = zip(paths, groups)
    rng.shuffle(paths_and_groups)
    for i, (path, group) in enumerate(paths_and_groups):
        source = {}
        source["path"] = path
        source["label"] = group
        sources.append(source)

    data["sources"] = sources

    return data

# Trainset
nb_controls = len(controls_splitted[0])
nb_patients = len(patients_splitted[0])
t1_paths = controls_splitted[0] + patients_splitted[0]
groups = [0] * nb_controls + [1] * nb_patients
json_trainset = build_json(t1_paths, groups, "PPMI-train", rng)
save_dict_to_json_file("PPMI_trainset.json", json_trainset)

# Validset
nb_controls = len(controls_splitted[1])
nb_patients = len(patients_splitted[1])
t1_paths = controls_splitted[1] + patients_splitted[1]
groups = [0] * nb_controls + [1] * nb_patients
json_validset = build_json(t1_paths, groups, "PPMI-valid", rng)
save_dict_to_json_file("PPMI_validset.json", json_validset)

# Testset
nb_controls = len(controls_splitted[2])
nb_patients = len(patients_splitted[2])
t1_paths = controls_splitted[2] + patients_splitted[2]
groups = [0] * nb_controls + [1] * nb_patients
json_testset = build_json(t1_paths, groups, "PPMI-test", rng)
save_dict_to_json_file("PPMI_testset.json", json_testset)

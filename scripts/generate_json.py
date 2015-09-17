#!/usr/bin/env python

from collections import OrderedDict
from itertools import izip_longest

import os
import json
import argparse


def buildArgsParser():
    DESCRIPTION = "Script to generate a json config file used by the brainsearch project."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('filenames', metavar="file", type=str, nargs="+", help='nifti files (.nii|nii.gz)')
    p.add_argument('--name', type=str, help="name of the json config file. Default: 'config'", default="config")
    p.add_argument('--masks', type=str, nargs="*", help="nifti files (.nii|nii.gz) containing a mask every `file`. Default: []")

    return p


def save_dict_to_json_file(path, dictionary):
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': ')))


def main(brain_manager=None):
    parser = buildArgsParser()
    args = parser.parse_args()

    sources = []
    for filename, mask in izip_longest(args.filenames, args.masks):
        subject = {}
        subject["path"] = os.path.abspath(filename)

        if mask is not None:
            subject["mask"] = os.path.abspath(mask)

        subject["name"] = os.path.basename(filename).split(".nii")[0]
        subject["id"] = int(subject["name"].split("_")[1])

        if subject["name"].split("_")[0].lower() == "control":
            subject["label"] = 0
        else:
            subject["label"] = 1

        sources.append(subject)

    config = OrderedDict()
    config["name"] = args.name
    config["type"] = "nifti"
    config["sources"] = sources

    save_dict_to_json_file(args.name + ".json", config)

if __name__ == '__main__':
    main()

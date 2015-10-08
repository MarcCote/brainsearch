#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from os.path import join as pjoin
import argparse

import nibabel as nib


def buildArgsParser():
    DESCRIPTION = "Script to create frankensteined brains."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('fa', type=str, help="fa image (.nii|nii.gz).")
    p.add_argument('--shape', type=str, help="size of the patch(es).", default="9,9,9")
    p.add_argument('--type', type=str, help="type of the patch(es).", choices={'brain', 'uniform'}, default="uniform")
    p.add_argument('--sigma', type=float, help="sigma (std) for Gaussian additive noise. Default: 0", default=0.)
    p.add_argument('--value', type=float, help="value of the patch(es) if uniform.", default=0.22)
    p.add_argument('--count', type=int, help="number of freaks to generate.", default=1)
    p.add_argument('--out', type=str, help="output folder", default="./")
    p.add_argument('--seed-pos', type=int, help="seed of the random generator used for the patch position", default=1234)
    p.add_argument('--seed-patch', type=int, help="seed of the random generator used for the patch values", default=1234)
    p.add_argument('-f', '--force', action="store_true", help="force overwrite")

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    print(args)

    fa = nib.load(args.fa)

    patch_shape = map(int, args.shape.split(','))

    # Patch type
    if args.type == "brain":
        pass
    elif args.type == "uniform":
        patch = args.value * np.ones(patch_shape)
    else:
        raise ValueError("Unknown patch type: {}".format(args.patch_type))

    # Add Gaussian additive noise.
    rng = np.random.RandomState(args.seed_patch)
    patch += args.sigma * rng.randn(*patch_shape).astype(fa.get_data().dtype)

    # Make sure the values of the patch are between 0 and 1.
    patch = np.minimum(patch, 1)
    patch = np.maximum(patch, 0)

    # Randomly choose the position of the patch's top-left corner.
    rng = np.random.RandomState(args.seed_pos)
    for i in range(args.count):
        fa_data = fa.get_data().copy()
        mask_data = np.zeros(fa_data.shape, dtype="int8")
        bigger_mask_data = np.zeros(fa_data.shape, dtype="int8")

        x = rng.randint(fa.shape[0])
        y = rng.randint(fa.shape[1])
        z = rng.randint(fa.shape[2])
        valid_seed = np.sum(fa_data[x:x+patch_shape[0], y:y+patch_shape[1], z:z+patch_shape[2]] == 0.0) < 5

        while not valid_seed:
            x = rng.randint(fa.shape[0])
            y = rng.randint(fa.shape[1])
            z = rng.randint(fa.shape[2])
            valid_seed = np.sum(fa_data[x:x+patch_shape[0], y:y+patch_shape[1], z:z+patch_shape[2]] == 0.0) < 5

        # Modify patch
        fa_data[x:x+patch_shape[0], y:y+patch_shape[1], z:z+patch_shape[2]] = patch
        mask_data[x:x+patch_shape[0], y:y+patch_shape[1], z:z+patch_shape[2]] = 1
        bigger_mask_data[max(0, x-10):min(x+patch_shape[0]+10, fa.shape[0]),
                         max(0, y-10):min(y+patch_shape[1]+10, fa.shape[1]),
                         max(0, z-10):min(z+patch_shape[2]+10, fa.shape[2])] = 1

        # Save the frankenstenized brain.
        filename = os.path.basename(args.fa).replace("control", "freak")
        splits = filename.split("_")
        filename = splits[0] + "_{0}{1}_{1}_".format(splits[1], i) + "_".join(splits[2:])

        if not os.path.isdir(args.out):
            os.mkdir(args.out)

        save_path = pjoin(args.out, filename)
        if not os.path.isfile(save_path) or args.force:
            #nii_fa_freak = nib.Nifti1Image(fa_data, fa.affine, header=fa.header, extra=fa.extra)
            nii_fa_freak = nib.Nifti1Image(fa_data, np.eye(4))
            nib.save(nii_fa_freak, save_path)
            #nii_mask_freak = nib.Nifti1Image(mask_data, fa.affine, header=fa.header, extra=fa.extra)
            nii_mask_freak = nib.Nifti1Image(mask_data, np.eye(4))
            nib.save(nii_mask_freak, save_path.split(".nii.gz")[0] + "_mask.nii.gz")
            #nii_bigger_mask_freak = nib.Nifti1Image(bigger_mask_data, fa.affine, header=fa.header, extra=fa.extra)
            nii_bigger_mask_freak = nib.Nifti1Image(bigger_mask_data, np.eye(4))
            nib.save(nii_bigger_mask_freak, save_path.split(".nii.gz")[0] + "_bigmask.nii.gz")
        else:
            print "File already existing. Use -f to override it."


if __name__ == '__main__':
    main()

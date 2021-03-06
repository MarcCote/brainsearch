#!/usr/bin/env python
from __future__ import division

import json
import numpy as np
import nibabel as nib

from brainsearch.brain_data import brain_data_factory
from brainsearch.utils import Timer2 as Timer

from brainsearch.brain_processing import BrainPipelineProcessing, BrainNormalization, BrainResampling

import argparse


def buildArgsParser():
    DESCRIPTION = "Script to generate a tmap and a pmap using a two-tailed hypothesis test of difference of means."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('configs', type=str, nargs="+", help='JSON file describing the data')

    p.add_argument('-r', dest="resampling_factor", type=float, help='resample image before processing', default=1.)
    p.add_argument('--norm', dest="do_normalization", action="store_true", help='perform histogram equalization')

    return p


def save_nifti(image, affine, name):
    nifti = nib.Nifti1Image(image, affine)
    nib.save(nifti, name)


def main(brain_manager=None):
    parser = buildArgsParser()
    args = parser.parse_args()

    # Build processing pipeline
    pipeline = BrainPipelineProcessing()
    if args.do_normalization:
        pipeline.add(BrainNormalization(type=0))
    if args.resampling_factor > 1:
        pipeline.add(BrainResampling(args.resampling_factor))

    #controls = defaultdict(lambda: [])
    #parkinsons = defaultdict(lambda: [])
    mean_controls = None
    mean_parkinsons = None
    nb_controls = 0
    nb_parkinsons = 0
    dtype = np.float32

    with Timer("Computing mean of samples"):
        for config in args.configs:
            config = json.load(open(config))
            brain_data = brain_data_factory(config, pipeline=pipeline)

            for brain in brain_data:
                if mean_controls is None and mean_parkinsons is None:
                    mean_controls = np.zeros_like(brain.image, dtype=dtype)
                    mean_parkinsons = np.zeros_like(brain.image, dtype=dtype)

                if brain.image.shape != mean_controls.shape or brain.image.shape != mean_parkinsons.shape:
                    print "Oups shapes not the same!"
                    from ipdb import set_trace as dbg
                    dbg()

                with Timer("Processing {}".format(brain.name)):
                    if brain.label == 0:
                        nb_controls += 1
                        mean_controls += brain.image
                    elif brain.label == 1:
                        nb_parkinsons += 1
                        mean_parkinsons += brain.image
                    else:
                        print "Unknown brain label: {}".format(brain.label)

        mean_controls /= nb_controls
        mean_parkinsons /= nb_parkinsons

    std_controls = np.zeros_like(mean_controls, dtype=dtype)
    std_parkinsons = np.zeros_like(mean_parkinsons, dtype=dtype)
    with Timer("Computing standard deviation of samples"):
        for config in args.configs:
            config = json.load(open(config))
            brain_data = brain_data_factory(config, pipeline=pipeline)

            for brain in brain_data:
                with Timer("Processing {}".format(brain.name)):
                    if brain.label == 0:
                        std_controls += (brain.image - mean_controls)**2
                    elif brain.label == 1:
                        std_parkinsons += (brain.image - mean_parkinsons)**2

        std_controls = np.sqrt(std_controls / (nb_controls-1))
        std_parkinsons = np.sqrt(std_parkinsons / (nb_parkinsons-1))

    s1 = std_controls
    n1 = nb_controls
    s2 = std_parkinsons
    n2 = nb_parkinsons

    # Compute the test statistic t
    stderror = np.sqrt((s1**2/n1) + (s2**2/n2))
    # The Null hypothesis : mu1 - mu2 = 0
    tmap = ((mean_parkinsons-mean_controls) - 0) / stderror
    tmap[stderror == 0] = 0  # Empty voxels

    # Compute p-value
    DF_numerator = (s1**2/n1 + s2**2/n2)**2
    DF_devisor = ((s1**2/n1)**2/(n1-1)) + ((s2**2/n2)**2/(n2-1))
    DF = DF_numerator // DF_devisor
    DF[DF_devisor == 0] = 0  # Empty voxels

    import scipy.stats as stat
    pmap = 2 * stat.t.cdf(-abs(tmap), DF)  # Two-tailed test, take twice the lower tail.
    pmap[np.isnan(pmap)] = 1  # Empty voxels
    save_nifti(tmap, brain.infos['affine'], 'tmap.nii.gz')
    save_nifti(pmap, brain.infos['affine'], 'pmap.nii.gz')
    save_nifti(1-pmap, brain.infos['affine'], 'inv_pmap.nii.gz')

if __name__ == '__main__':
    main()

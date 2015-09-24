#!/usr/bin/env python
from __future__ import division

import os
from os.path import join as pjoin

import json
import time
import numpy as np
import pylab as plt
import itertools
import nibabel as nib

from itertools import izip, chain
import brainsearch.vizu as vizu

from brainsearch.utils import Timer
import brainsearch.utils as brainutil
#from brainsearch.imagespeed import blockify
from brainsearch.brain_database import BrainDatabaseManager
from brainsearch.brain_data import brain_data_factory

import nearpy
from nearpy.hashes import LocalitySensitiveHashing, PCAHashing, SpectralHashing
from nearpy.distances import EuclideanDistance
from nearpy.filters import SortedFilter, NearestFilter, DistanceThresholdFilter
from nearpy.utils import chunk, ichunk

from brainsearch.brain_processing import BrainPipelineProcessing, BrainNormalization, BrainResampling


def hist_prop(p, bins=25, P0=None, show=True, *args, **kwargs):
    if show:
        plt.figure()

    if type(bins) is int:
        plt.hist(p, bins=bins, *args, **kwargs)
    elif bins.upper() == "FD":
        # Plot histogram using Freedman & Diaconis rule.
        # http://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
        IQR = lambda x: np.subtract(*np.percentile(x, [75, 25]))
        plt.hist(p, bins=1./(2*IQR(p)*len(p)**(-1/3.)), normed=True, *args, **kwargs)

    if P0 is not None:
        plt.plot([P0]*2, [0, 1], 'r', linewidth=2)

    if show:
        plt.show()


def two_tailed_test_of_population_proportion(P0, p, n):
    """
    P0: Hypothesized value of the true population proportion
    p: sample proportion
    n: sample size
    """
    # Compute test statistic
    # http://stattrek.com/hypothesis-test/proportion.aspx
    sigma = np.sqrt(P0*(1-P0)/n)
    z_statistic = (p - P0) / sigma

    # Compute p-value
    import scipy.stats as stat
    pvalue = 2 * stat.norm.cdf(-abs(z_statistic))  # Two-tailed test, take twice the lower tail.

    return z_statistic, pvalue


def proportion_map(positives, negatives, ratio_pos):
    N = positives + negatives
    proportions = positives / N
    return np.nan_to_num(proportions)


def hack_map(positives, negatives, ratio_pos):
    ratio_neg = 1. - ratio_pos
    N = positives + negatives

    # Assume binary classification for now
    nb_neg = 1./(1. + negatives)
    nb_pos = 1./(1. + positives)

    pos = nb_pos * ratio_neg
    neg = nb_neg * ratio_pos
    m = (pos-neg) / N
    idx = m > 0
    m[idx] /= ratio_neg
    m[np.bitwise_not(idx)] /= ratio_pos

    return (m+1)/2.


def save_nifti(image, affine, name):
    nifti = nib.Nifti1Image(image, affine)
    nib.save(nifti, name)


def main(brain_manager=None):
    if brain_manager is None:
        brain_manager = BrainDatabaseManager(args.storage)

    # Build processing pipeline
    pipeline = BrainPipelineProcessing()
    if args.do_normalization:
        pipeline.add(BrainNormalization(type=0))
    if args.resampling_factor > 1:
        pipeline.add(BrainResampling(args.resampling_factor))


def list(brain_manager, name, verbose=False, check_integrity=False):
    def print_info(name, brain_db):
        print name
        #print "\tMetadata:", brain_db.metadata
        print "\tPatch size:", brain_db.metadata["patch"].shape
        print "\tHashes:", map(str, brain_db.engine.lshashes)
        if verbose:
            labels_counts = ["{}: {:,}".format(i, label_count) for i, label_count in enumerate(brain_db.labels_count(check_integrity=check_integrity))]
            print "\tLabels: {" + "; ".join(labels_counts) + "}"
            print "\tPatches: {:,}".format(brain_db.nb_patches(check_integrity=check_integrity))
            print "\tBuckets: {:,}".format(brain_db.nb_buckets(check_integrity=check_integrity))

    if name in brain_manager:
        print_info(name, brain_manager[name])
    else:
        print "{} available brain databases: ".format(len(brain_manager.brain_databases_names))
        for name in brain_manager.brain_databases_names:
            try:
                print_info(name, brain_manager[name])
                print ""
            except:
                import traceback
                traceback.print_exc()
                print "*Brain database '{}' is corrupted!*\n".format(name)


def clear(brain_manager, names, force=False):
    start = time.time()
    if len(names) == 0:
        print "Clearing all"
        brain_manager.remove_all_brain_databases(force)
    else:
        for name in names:
            brain_db = brain_manager[name]
            if brain_db is None:
                raise ValueError("Unexisting brain database: " + name)

            print "Clearing", name
            brain_manager.remove_brain_database(brain_db, force)

    print "Done in {:.2f} sec.".format(time.time()-start)


def hashing_factory(hashtype, dimension, nbits, **kwargs):
    if hashtype.upper() == "SH":
        hash_name = "SH{nbits}".format(nbits=nbits)
        return SpectralHashing(hash_name,
                               nbits=nbits,
                               dimension=dimension,
                               trainset=kwargs['trainset'],
                               pca_pkl=kwargs['pca_pkl'],
                               bounds_pkl=kwargs['bounds_pkl'])

    elif hashtype.upper() == "PCA":
        hash_name = "PCAH{nbits}".format(nbits=nbits)
        return PCAHashing(hash_name,
                          nbits=nbits,
                          dimension=dimension,
                          trainset=kwargs['trainset'],
                          pca_pkl=kwargs['pca_pkl'])

    elif hashtype.upper() == "LSH":
        hash_name = "LSH{nbits}".format(nbits=nbits)
        return LocalitySensitiveHashing(hash_name,
                                        nbits=nbits,
                                        dimension=dimension)

    # if args.LSH_PCA is not None:
    #     for nb_projections in args.LSH_PCA:
    #         hash_name = "LSH_PCA{nb_projections}".format(nb_projections=nb_projections)
    #         hashes.append(RandomPCABinaryProjections(hash_name, projection_count=nb_projections, dimension=np.prod(patch_shape), trainset=_get_all_patches(), pkl=args.pkl))

    raise ValueError("Unknown hashing method: {}".format(hashtype))


def init(brain_manager, name, patch_shape, hashing):
    metadata = {b"patch": {"dtype": np.dtype(np.float32).str, "shape": patch_shape},
                b"label": {"dtype": np.dtype(np.int8).str, "shape": (1,)},
                b"id": {"dtype": np.dtype(np.int32).str, "shape": (1,)},
                b"position": {"dtype": np.dtype(np.int32).str, "shape": (len(patch_shape),)},
                }

    brain_manager.new_brain_database(name, hashing, metadata)


def add(brain_manager, name, brain_data, min_nonempty=0, spatial_weight=0.):
    brain_db = brain_manager[name]
    if brain_db is None:
        raise ValueError("Unexisting brain database: " + name)

    patch_shape = tuple(brain_db.metadata['patch'].shape)

    print 'Inserting...'
    nb_elements_total = 0
    start = time.time()
    for brain_id, brain in enumerate(brain_data):
        start_brain = time.time()
        with Timer("  Extracting"):
            brain_patches = brain.extract_patches(patch_shape, min_nonempty=min_nonempty)
            vectors = brain_patches.create_vectors(spatial_weight=spatial_weight)

        hashkeys = brain_db.insert(vectors, brain_patches)

        print "ID: {0} (label:{3}), {1:,} patches in {2:.2f} sec.".format(brain_id, len(hashkeys), time.time()-start_brain, brain.label)
        nb_elements_total += len(hashkeys)

    print "Inserted {0:,} patches ({1} brains) in {2:.2f} sec.".format(nb_elements_total, brain_id+1, time.time()-start)


def check(brain_manager, name, spatial_weight=0.):
    brain_db = brain_manager[name]
    if brain_db is None:
        raise ValueError("Unexisting brain database: " + name)

    # Simply report stats about buckets size.
    with Timer('Counting'):
        sizes, bucketkeys = brain_db.buckets_size()
        sizes = np.array(sizes)

    print "Counted {1:,} candidates for {0:,} buckets".format(len(sizes), sum(sizes))
    print "Avg. candidates per bucket: {0:.2f}".format(np.mean(sizes))
    print "Std. candidates per bucket: {0:.2f}".format(np.std(sizes))
    print "Min. candidates per bucket: {0:,}".format(np.min(sizes))
    print "Max. candidates per bucket: {0:,}".format(np.max(sizes))
    print "Sum_bucket |bucket|*(|bucket|-1): {0:,}".format(np.sum(sizes*(sizes-1)))

    std_voxels = []
    with Timer('\nEvaluating std. of voxels values in a bucket'):
        bucket_sorted_indices = np.argsort(sizes)[::-1]
        for idx in bucket_sorted_indices:
            if sizes[idx] < 100:
                break

            patches = brain_db.engine.storage.retrieve([bucketkeys[idx]], attribute=brain_db.metadata['patch'])[0]
            std_voxels.append(np.std(patches, axis=0))
            #print sizes[idx], std_voxels[-1].flatten()

    from ipdb import set_trace as dbg
    dbg()
    rng = np.random.RandomState(42)

    NB_PAIRS = 1000
    avg_spatial_distances = []
    std_positions = []
    with Timer('\nEvaluating avg. spatial distance between {} patches'.format(NB_PAIRS)):
        bucket_sorted_indices = np.argsort(sizes)[::-1]
        for idx in bucket_sorted_indices:
            if sizes[idx] < 100:
                break

            positions = brain_db.engine.storage.retrieve([bucketkeys[idx]], attribute=brain_db.metadata['position'])[0]
            std_positions.append(np.std(positions, axis=0))

            indices = np.arange(len(positions))
            pairs = rng.choice(indices, size=(min(len(indices)//2, NB_PAIRS), 2), replace=False)
            selected_positions = positions[pairs]
            distance = np.sqrt(np.sum((selected_positions[:, -1] - selected_positions[:, 0])**2, axis=1))
            avg_spatial_distances.append(np.mean(distance))

    print "Avg. spatial distance per bucket: {0:.2f}".format(np.mean(avg_spatial_distances))
    print "Std. spatial distance per bucket: {0:.2f}".format(np.std(avg_spatial_distances))
    print "Avg. of position std.: {}".format(np.mean(std_positions, axis=0))

    plt.clf()
    plt.hist(sizes, bins=np.logspace(0, np.log10(np.max(sizes))), log=True)
    plt.xlabel('Bucket sizes')
    plt.ylabel('Count')
    plt.xscale('log')
    #plt.show()

    FIGURES_FOLDER = './figs'
    if not os.path.isdir(FIGURES_FOLDER):
        os.mkdir(FIGURES_FOLDER)

    plt.savefig(pjoin(FIGURES_FOLDER, name), bbox_inches='tight')

    #brain_db.show_large_buckets(sizes, bucketkeys, spatial_weight)


def create_map(brain_manager, name, brain_data, K=100, threshold=np.inf, min_nonempty=0, spatial_weight=0.):
    brain_db = brain_manager[name.strip("/").split("/")[-1]]
    if brain_db is None:
        raise ValueError("Unexisting brain database: " + name)

    patch_shape = brain_db.metadata['patch'].shape

    brain_db.engine.distance = nearpy.distances.EuclideanDistance(brain_db.metadata['patch'])
    #brain_db.engine.distance = nearpy.distances.CorrelationDistance(brain_db.metadata['patch'])

    # TODO: find how to compute a good threshood :/ ?!?
    brain_db.engine.filters = [DistanceThresholdFilter(threshold), NearestFilter(K)]

    half_patch_size = np.array(patch_shape) // 2

    print "Found {} brains to map".format(len(brain_data))
    for i, brain in enumerate(brain_data):
        print "Mapping {}...".format(brain.name)
        brain_patches = brain.extract_patches(patch_shape, min_nonempty=min_nonempty)
        vectors = brain_patches.create_vectors(spatial_weight=spatial_weight)

        # Position of extracted patches represent to top left corner.
        center_positions = brain_patches.positions + half_patch_size

        nids = -1 * np.ones((len(brain_patches), K), dtype=np.uint8)  # No more than 256, okay for now,
        nlabels = -1 * np.ones((len(brain_patches), K), dtype=np.uint8)
        #ndists = -1 * np.ones((len(brain_patches), K), dtype=np.float32)
        #npositions = -1 * np.ones((len(brain_patches), K, 3), dtype=np.uint16)

        start_brain = time.time()
        for patch_id, neighbors in brain_db.get_neighbors(vectors, brain_patches.patches, attributes=["id", "label"]):
            nlabels[patch_id, :len(neighbors['label'])] = neighbors['label'].flatten()
            nids[patch_id, :len(neighbors['id'])] = neighbors['id'].flatten()
            #ndists[patch_id, :len(neighbors['dist'])] = neighbors['dist'].flatten()
            #npositions[patch_id, :len(neighbors['position']), :] = neighbors['position']

        print "{4}. Brain #{0} ({3:,} patches) found {1:,} neighbors in {2:.2f} sec.".format(brain.id, np.sum(nlabels != -1), time.time()-start_brain, len(brain_patches), i)
        print "Patches with no neighbors: {:,}".format(np.all(nlabels == -1, axis=1).sum())

        ## Generate map of p-values ##

        # Use leave-one-out strategy, i.e. do not use neighbors patches coming from the query brain.
        control = np.sum(np.logical_and(nlabels == 0, nids != brain.id), axis=1)
        parkinson = np.sum(np.logical_and(nlabels == 1, nids != brain.id), axis=1)

        # Weight the proportion by the distance of the query patch from neighbors patch
        #control = np.sum(np.exp(-ndists) * np.logical_and(nlabels == 0, nids != brain.id), axis=1)
        #parkinson = np.sum(np.exp(-ndists) * np.logical_and(nlabels == 1, nids != brain.id), axis=1)
        #control = np.sum((1-ndists) * np.logical_and(nlabels == 0, nids != brain.id), axis=1)
        #parkinson = np.sum((1-ndists) * np.logical_and(nlabels == 1, nids != brain.id), axis=1)

        P0 = brain_db.label_proportions()[1]  # Hypothesized population proportion
        p = np.nan_to_num(parkinson / (parkinson+control))  # sample proportion
        n = np.sum(nlabels != -1, axis=1)     # sample size

        z_statistic, pvalue = two_tailed_test_of_population_proportion(P0, p, n)

        zmap = np.zeros_like(brain.image, dtype=np.float32)
        pmap = np.ones_like(brain.image, dtype=np.float32)

        # Patches composite z-scores
        for z in range(patch_shape[2]):
            for y in range(patch_shape[1]):
                for x in range(patch_shape[0]):
                    pos = brain_patches.positions + np.array((x, y, z))
                    zmap[zip(*pos)] += z_statistic

        zmap[zip(*center_positions)] /= np.sqrt(np.prod(patch_shape))

        #zmap[zip(*center_positions)] = z_statistic
        pmap[zip(*center_positions)] = pvalue
        zmap[np.isnan(zmap)] = 0.
        pmap[np.isnan(pmap)] = 1.

        results_folder = pjoin('.', 'results', brain_db.name, brain_data.name)
        if not os.path.isdir(results_folder):
            os.makedirs(results_folder)

        save_nifti(brain.image, brain.infos['affine'], pjoin(results_folder, "{}.nii.gz".format(brain.name)))
        save_nifti(pmap, brain.infos['affine'], pjoin(results_folder, "{}_pmap.nii.gz".format(brain.name)))
        save_nifti(zmap, brain.infos['affine'], pjoin(results_folder, "{}_zmap.nii.gz".format(brain.name)))
        #np.savez(pjoin(results_folder, name), dists=ndists, labels=nlabels, ids=nids, positions=npositions, voxels_positions=center_positions)


def create_proximity_map(brain_manager, name, brain_data, K=100, threshold=np.inf, min_nonempty=0, spatial_weight=0.):
    brain_db = brain_manager[name.strip("/").split("/")[-1]]
    if brain_db is None:
        raise ValueError("Unexisting brain database: " + name)

    patch_shape = brain_db.metadata['patch'].shape

    brain_db.engine.distance = EuclideanDistance(brain_db.metadata['patch'])

    # TODO: find how to compute a good threshood :/ ?!?
    brain_db.engine.filters = [DistanceThresholdFilter(threshold), NearestFilter(K)]
    half_patch_size = np.array(patch_shape) // 2

    print "Found {} brains/regions for wich to compute a proximity-map".format(len(brain_data))
    for i, brain in enumerate(brain_data):
        print "Mapping {}...".format(brain.name)
        brain_patches = brain.extract_patches(patch_shape, min_nonempty=min_nonempty)
        vectors = brain_patches.create_vectors(spatial_weight=spatial_weight)

        #nids = -1 * np.ones((len(brain_patches), K), dtype=np.uint8)  # No more than 256, okay for now,
        #nlabels = -1 * np.ones((len(brain_patches), K), dtype=np.uint8)
        ndists = -1 * np.ones((len(brain_patches), K), dtype=np.float32)
        npositions = -1 * np.ones((len(brain_patches), K, 3), dtype=np.uint16)

        center_positions = brain_patches.positions + half_patch_size

        start_brain = time.time()
        for patch_id, neighbors in brain_db.get_neighbors(vectors, brain_patches.patches, attributes=["id", "label", "position"]):
        #for patch_id, neighbors in brain_db.get_neighbors(vectors, brain_patches.patches, attributes=["label"]):
            ndists[patch_id, :len(neighbors['dist'])] = neighbors['dist'].flatten()
            #nlabels[patch_id, :len(neighbors['label'])] = neighbors['label'].flatten()
            #nids[patch_id, :len(neighbors['id'])] = neighbors['id'].flatten()
            npositions[patch_id, :len(neighbors['position']), :] = neighbors['position']

            #if np.any(np.all(neighbors['position']+half_patch_size == (130, 41, 34), axis=1)):
            #  from ipdb import set_trace; set_trace()

        print "{4}. Brain #{0} ({3:,} patches) found {1:,} neighbors in {2:.2f} sec.".format(brain.id, np.sum(ndists != -1), time.time()-start_brain, len(brain_patches), i)
        print "Patches with no neighbors: {:,}".format(np.all(ndists == -1, axis=1).sum())

        ## Generate proximity-map ##
        positions = npositions.reshape((-1, 3))
        idx_to_keep = np.where(positions[:, 0] != -1)[0]
        positions = positions[idx_to_keep]
        distances = ndists.flatten()
        distances = distances[idx_to_keep]

        # Position of extracted patches represent to top left corner.
        center_positions = positions + half_patch_size

        #proxmap = np.nan * np.ones_like(brain.image, dtype=int)
        proxmap = np.zeros_like(brain.image, dtype=np.float32)
        for (x, y, z), dist in zip(center_positions, distances):
            #proxmap[x, y, z] += 1
            proxmap[x, y, z] += 1-dist
            #proxmap[x, y, z] += np.exp(-20000*dist)

        results_folder = pjoin('.', 'results', brain_db.name, brain_data.name)
        if not os.path.isdir(results_folder):
            os.makedirs(results_folder)

        save_nifti(brain.image, brain.infos['affine'], pjoin(results_folder, "{}.nii.gz".format(brain.name)))
        save_nifti(proxmap, brain.infos['affine'], pjoin(results_folder, "{}_proxmap.nii.gz".format(brain.name)))


# def vizu():
#     from brainsearch.vizu_chaco import NoisyBrainsearchViewer

#     brain_db = brain_manager[args.name]
#     if brain_db is None:
#         raise ValueError("Unexisting brain database: " + args.name)

#     patch_shape = brain_db.metadata['patch'].shape
#     config = json.load(open(args.config))
#     brain_data = brain_data_factory(config, pipeline=pipeline)

#     for brain_id, brain in enumerate(brain_data):
#         print 'Viewing brain #{0} (label: {1})'.format(brain_id, brain.label)
#         patches, positions = brain.extract_patches(patch_shape, min_nonempty=args.min_nonempty, with_positions=True)

#         query = {'patches': patches,
#                  'positions': positions,
#                  'patch_size': patch_shape}
#         viewer = NoisyBrainsearchViewer(query, brain_db.engine, brain_voxels=brain.image)
#         viewer.configure_traits()


# def eval():
#     brain_database = brain_manager[args.name]
#     if brain_database is None:
#         raise ValueError("Unexisting brain database: " + args.name)

#     patch_shape = tuple(brain_database.config['shape'])
#     config = json.load(open(args.config))
#     brain_data = brain_data_factory(config)

#     print 'Evaluating...'

#     def majority_vote(candidates):
#         return np.argmax(np.mean([np.array(c['data']['target']) for c in candidates], axis=0))

#     def weighted_majority_vote(candidates):
#         votes = [np.exp(-c['dist']) * np.array(c['data']['target']) for c in candidates]
#         return np.argmax(np.mean(votes, axis=0))

#     brain_database.engine.distance = EuclideanDistance()

#     #neighbors = []
#     nb_neighbors = 0
#     start = time.time()
#     nb_success = 0.0
#     nb_patches = 0
#     for brain_id, (brain, label) in enumerate(brain_data):
#         patches_and_pos = get_patches(brain, patch_shape=patch_shape, min_nonempty=args.min_nonempty)
#         patches = flattenize((patch for patch, pos in patches_and_pos))

#         start_brain = time.time()
#         neighbors_per_patch = brain_database.query(patches)
#         nb_patches += len(neighbors_per_patch)
#         brain_neighbors = list(chain(*neighbors_per_patch))
#         print "Brain #{0} ({3:,} patches), found {1:,} neighbors in {2:.2f} sec.".format(brain_id, len(brain_neighbors), time.time()-start_brain, len(neighbors_per_patch))

#         #prediction = weighted_majority_vote(brain_neighbors)
#         prediction = majority_vote(brain_neighbors)
#         nb_success += prediction == np.argmax(label)

#         nb_neighbors += len(brain_neighbors)
#         del brain_neighbors
#         #neighbors.extend(brain_neighbors)

#     nb_brains = brain_id + 1
#     print "Found a total of {0:,} neighbors for {1} brains ({3:,} patches) in {2:.2f} sec.".format(nb_neighbors, nb_brains, time.time()-start, nb_patches)
#     print "Classification error: {:2.2f}%".format(100 * (1. - nb_success/nb_brains))

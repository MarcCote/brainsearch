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

import brainsearch.utils as brainutil
#from brainsearch.imagespeed import blockify
from brainsearch.brain_database import BrainDatabaseManager
from brainsearch.brain_data import brain_data_factory

from nearpy.hashes import RandomBinaryProjections, RandomPCABinaryProjections, PCABinaryProjections, SpectralHashing
from nearpy.distances import EuclideanDistance
from nearpy.filters import NearestFilter, DistanceThresholdFilter
from nearpy.utils import chunk, ichunk

from brainsearch.brain_processing import BrainPipelineProcessing, BrainNormalization, BrainResampling

import argparse

#PORT = 4242
PORT = 6379
OFFSET = 0.01


def build_subcommand_list(subparser):
    DESCRIPTION = "List available brain databases."

    p = subparser.add_parser("list",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('name', type=str, nargs='?', help='name of the brain database')
    p.add_argument('-v', action='store_true', help='display more information about brain databases')
    p.add_argument('-f', action='store_true', help='check integrity of brain databases')


def build_subcommand_clear(subparser):
    DESCRIPTION = "Clear brain databases."

    p = subparser.add_parser("clear",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('names', metavar="name", type=str, nargs="*", help='name of the brain database to delete')
    p.add_argument('-f', action='store_true', help='clear also metadata')


def build_subcommand_init(subparser):
    DESCRIPTION = "Build a new brain database (nearpy's engine)."

    p = subparser.add_parser("init",
                             description=DESCRIPTION,
                             help=DESCRIPTION)

    p.add_argument('name', type=str, help='name of the brain database')
    p.add_argument('shape', metavar="X,Y,...", type=str, help="data's shape or patch shape")
    p.add_argument('--LSH', metavar="N", type=int, nargs="+", help='numbers of random projections')
    p.add_argument('--LSH_PCA', metavar="N", type=int, nargs="+", help='numbers of random projections in PCA space')
    p.add_argument('--PCA', metavar="K", type=int, nargs="+", help='use K eigenvectors')
    p.add_argument('--SH', metavar="K", type=int, nargs="+", help='length of hash codes generated by Spectral Hashing')
    p.add_argument('--trainset', type=str, help='JSON file use to "train" PCA')
    p.add_argument('--pca', type=str, help='pickle file containing the PCA information of the data')
    p.add_argument('--bounds', type=str, help='pickle file containing the bounds used by spectral hashing')


def build_subcommand_add(subparser):
    DESCRIPTION = "Add data to an existing brain database."

    p = subparser.add_parser("add",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('name', type=str, help='name of the brain database')
    p.add_argument('config', type=str, help='contained in a JSON file')


def build_subcommand_eval(subparser):
    DESCRIPTION = "Evaluate data given an existing brain database."

    p = subparser.add_parser("eval",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('name', type=str, help='name of the brain database')
    p.add_argument('config', type=str, help='contained in a JSON file')
    p.add_argument('-k', type=int, help='consider at most K nearest-neighbors')


def build_subcommand_map(subparser):
    DESCRIPTION = "Create a color map for a brain given an existing brain database."

    p = subparser.add_parser("map",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('name', type=str, help='name of the brain database')
    p.add_argument('config', type=str, help='contained in a JSON file')
    p.add_argument('-k', type=int, help='consider at most K nearest-neighbors', default=100)
    p.add_argument('--prefix', type=str, help="prefix for the name of the results files", default="")
    p.add_argument('--radius', type=int, help="only look at neighbors within a certain radius")


def build_subcommand_vizu(subparser):
    DESCRIPTION = "Run some vizu for a brain given an existing brain database."

    p = subparser.add_parser("vizu",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('name', type=str, help='name of the brain database')
    p.add_argument('config', type=str, help='contained in a JSON file')


def build_subcommand_check(subparser):
    DESCRIPTION = "Check candidates distribution given an existing brain database."

    p = subparser.add_parser("check",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('name', type=str, help='name of the brain database')
    p.add_argument('config', type=str, nargs='?', help='contained in a JSON file')
    #p.add_argument('-m', dest="min_nonempty", type=int, help='consider only patches having this minimum number of non-empty voxels')


def buildArgsParser():
    DESCRIPTION = "Script to perform brain searches."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('--storage', type=str, default="redis", help='which storage to use: redis, memory, file')

    p.add_argument('--use_spatial_code', action='store_true', help='include spatial position of a patch in hashcode')
    p.add_argument('-m', dest="min_nonempty", type=int, help='consider only patches having this minimum number of non-empty voxels')
    p.add_argument('--skip', metavar="N", type=int, help='skip N images', default=0)
    p.add_argument('-r', dest="resampling_factor", type=float, help='resample image before processing', default=1.)
    p.add_argument('--norm', dest="do_normalization", action="store_true", help='perform histogram equalization')

    subparser = p.add_subparsers(title="brain_search commands", metavar="", dest="command")
    build_subcommand_list(subparser)
    build_subcommand_init(subparser)
    build_subcommand_add(subparser)
    build_subcommand_eval(subparser)
    build_subcommand_map(subparser)
    build_subcommand_vizu(subparser)
    build_subcommand_check(subparser)
    build_subcommand_clear(subparser)

    return p


def proportion_test_map(positives, negatives, ratio_pos):
    P = ratio_pos
    N = positives + negatives
    voxel_std = np.sqrt(P*(1-P)/N)
    probs = positives / N
    Z = (probs-P) / voxel_std
    return Z


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
    parser = buildArgsParser()
    args = parser.parse_args()

    if brain_manager is None:
        brain_manager = BrainDatabaseManager(args.storage)

    # Build processing pipeline
    pipeline = BrainPipelineProcessing()
    if args.do_normalization:
        pipeline.add(BrainNormalization(type=0))
    if args.resampling_factor > 1:
        pipeline.add(BrainResampling(args.resampling_factor))

    if args.command == "list":
        def print_info(name, brain_db):
            print name
            #print "\tMetadata:", brain_db.metadata
            print "\tPatch size:", brain_db.metadata["patch"].shape
            print "\tHashes:", map(str, brain_db.engine.lshashes)
            if args.v:
                labels_counts = ["{}: {:,}".format(i, label_count) for i, label_count in enumerate(brain_db.labels_count(check_integrity=args.f))]
                print "\tLabels: {" + "; ".join(labels_counts) + "}"
                print "\tPatches: {:,}".format(brain_db.nb_patches(check_integrity=args.f))
                print "\tBuckets: {:,}".format(brain_db.nb_buckets())

        if args.name in brain_manager:
            print_info(args.name, brain_manager[args.name])
        else:
            print "Available brain databases: "
            for name, brain_db in brain_manager.brain_databases.items():
                try:
                    print_info(name, brain_db)
                except:
                    import traceback
                    traceback.print_exc()
                    print "*Brain database '{}' is corrupted!*\n".format(name)

    elif args.command == "clear":
        start = time.time()
        if len(args.names) == 0:
            print "Clearing all"
            brain_manager.remove_all_brain_databases(args.f)
        else:
            for name in args.names:
                brain_db = brain_manager[name]
                if brain_db is None:
                    raise ValueError("Unexisting brain database: " + name)

                print "Clearing", name
                brain_manager.remove_brain_database(brain_db, args.f)

        print "Done in {:.2f} sec.".format(time.time()-start)

    elif args.command == "init":
        print "Creating brain database {}...".format(args.name)
        if args.name in brain_manager:
            print ("This database already exists. Please use command "
                   "'brain_search.py --storage {} clear -f {}' before.".format(args.storage, args.name))
            exit()

        start = time.time()
        patch_shape = tuple(map(int, args.shape.split(",")))

        def _get_all_patches():
            config = json.load(open(args.trainset))
            brain_data = brain_data_factory(config, pipeline=pipeline)
            for brain_id, brain in enumerate(brain_data):
                print "ID: {0}/{1}".format(brain_id, len(brain_data))
                #if brain_id == 10: break  # TODO remove
                patches, positions = brain.extract_patches(patch_shape, min_nonempty=args.min_nonempty, with_positions=True)
                vectors = patches.reshape((-1, np.prod(patch_shape)))

                if args.use_spatial_code:
                    # Normalize position
                    pos_normalized = positions / np.array(brain.infos['img_shape'], dtype="float32")
                    vectors = np.c_[pos_normalized, vectors]

                yield vectors

        dimension = np.prod(patch_shape)
        if args.use_spatial_code:
            dimension += len(patch_shape)

        hashes = []
        if args.SH is not None:
            for nbits in args.SH:
                hash_name = "SH{nbits}".format(nbits=nbits)
                hashes.append(SpectralHashing(hash_name, nbits=nbits, dimension=dimension,
                                              trainset=_get_all_patches,
                                              pkl=args.pkl, bounds=args.bounds))

        if args.LSH_PCA is not None:
            for nb_projections in args.LSH_PCA:
                hash_name = "LSH_PCA{nb_projections}".format(nb_projections=nb_projections)
                hashes.append(RandomPCABinaryProjections(hash_name, projection_count=nb_projections, dimension=np.prod(patch_shape), trainset=_get_all_patches(), pkl=args.pkl))

        if args.PCA is not None:
            for nbits in args.PCA:
                hash_name = "PCA{nbits}".format(nbits=nbits)
                hashes.append(PCABinaryProjections(hash_name, dimension=np.prod(patch_shape),
                              trainset=_get_all_patches, nbits=nbits, pkl=args.pkl))

        if args.LSH is not None:
            for nb_projections in args.LSH:
                    hash_name = "LSH{nb_projections}".format(nb_projections=nb_projections)
                    hashes.append(RandomBinaryProjections(hash_name, nb_projections, dimension=np.prod(patch_shape)))

        metadata = {b"patch": {"dtype": np.dtype(np.float32).str, "shape": patch_shape},
                    b"label": {"dtype": np.dtype(np.int8).str, "shape": (1,)},
                    b"id": {"dtype": np.dtype(np.int32).str, "shape": (1,)},
                    b"position": {"dtype": np.dtype(np.int32).str, "shape": (len(patch_shape),)},
                    }
        brain_manager.new_brain_database(args.name, hashes[0], metadata)
        print "Created in {0:.2f} sec.".format(time.time()-start)

    elif args.command == "add":
        brain_db = brain_manager[args.name]
        if brain_db is None:
            raise ValueError("Unexisting brain database: " + args.name)

        patch_shape = tuple(brain_db.metadata['patch'].shape)
        config = json.load(open(args.config))
        brain_data = brain_data_factory(config, skip=args.skip, pipeline=pipeline)

        print 'Inserting...'
        nb_elements_total = 0
        start = time.time()
        for brain_id, brain in enumerate(brain_data, start=args.skip):
            #if brain_id == 10: break  # TODO remove

            start_brain = time.time()
            #image = brain.process(args.resampling_factor, args.do_normalization)
            #image, affine = brain.resample(args.resampling_factor)
            #patches, datainfo = get_patches_with_info(image, brain_id, brain.label, patch_shape=patch_shape, min_nonempty=args.min_nonempty)

            start_extracting = time.time()
            patches, datainfo = brain.extract_patches(patch_shape, min_nonempty=args.min_nonempty, with_info=True)
            print "extracting: {:.2f}".format(time.time()-start_extracting)

            vectors = patches.reshape((-1, np.prod(patch_shape)))
            if args.use_spatial_code:
                # Normalize position
                pos_normalized = datainfo["position"] / np.array(brain.infos['img_shape'], dtype="float32")
                pos_normalized = pos_normalized.astype("float32")
                vectors = np.c_[pos_normalized, vectors]

            hashkeys = brain_db.insert(vectors, datainfo["patch"], datainfo["label"], datainfo["position"], datainfo["id"])

            print "ID: {0} (label:{3}), {1:,} patches in {2:.2f} sec.".format(brain_id, len(hashkeys), time.time()-start_brain, brain.label)
            nb_elements_total += len(hashkeys)

        print "Inserted {0:,} patches ({1} brains) in {2:.2f} sec.".format(nb_elements_total, brain_id+1, time.time()-start)

    elif args.command == "check":
        brain_db = brain_manager[args.name]
        if brain_db is None:
            raise ValueError("Unexisting brain database: " + args.name)

        if args.config is None:
            # Simply report stats about buckets size.
            print 'Counting...'
            start = time.time()
            sizes, bucketkeys = brain_db.buckets_size()
            print "Counted {2:,} candidates for {0:,} buckets in {1:.2f} sec.".format(len(sizes), time.time()-start, sum(sizes))
            print "Avg. candidates per bucket: {0:.2f}".format(np.mean(sizes))
            print "Std. candidates per bucket: {0:.2f}".format(np.std(sizes))
            print "Min. candidates per bucket: {0:,}".format(np.min(sizes))
            print "Max. candidates per bucket: {0:,}".format(np.max(sizes))

            plt.hist(sizes, bins=np.logspace(0, np.log10(np.max(sizes))), log=True)
            plt.xscale('log')
            plt.show()

            brain_db.show_large_buckets(sizes, bucketkeys, args.use_spatial_code)
        else:
            patch_shape = brain_db.metadata['patch'].shape
            config = json.load(open(args.config))
            brain_data = brain_data_factory(config)

            print 'Counting...'
            candidate_count_per_brain = []
            start = time.time()
            for brain_id, (brain, _) in enumerate(brain_data):
                patches = brain.extract_patches(patch_shape, min_nonempty=args.min_nonempty)
                #patches, positions = get_patches(brain, patch_shape=patch_shape, min_nonempty=args.min_nonempty)

                start_brain = time.time()
                candidate_count_per_patch = brain_db.candidate_count(patches)
                candidate_count_per_brain.append(candidate_count_per_patch)
                print "ID: {0}, {1:,} patches ({3:,} candidates) in {2:.2f} sec.".format(brain_id, len(patches), time.time()-start_brain, sum(candidate_count_per_patch))

            candidate_count_per_patch = list(itertools.chain(*candidate_count_per_brain))
            print "Counted {2:,} candidates for {0:,} patches in {1:.2f} sec.".format(len(candidate_count_per_patch), time.time()-start, sum(map(sum, candidate_count_per_brain)))
            print "Avg. candidates per patch: {0:.2f}".format(np.mean(candidate_count_per_patch))
            print "Min. candidates per patch: {0:,}".format(np.min(candidate_count_per_patch))
            print "Max. candidates per patch: {0:,}".format(np.max(candidate_count_per_patch))

            plt.hist(candidate_count_per_patch, bins=np.logspace(0, np.log10(np.max(candidate_count_per_patch))), log=True)
            plt.show()

    elif args.command == "eval":
        brain_database = brain_manager[args.name]
        if brain_database is None:
            raise ValueError("Unexisting brain database: " + args.name)

        patch_shape = tuple(brain_database.config['shape'])
        config = json.load(open(args.config))
        brain_data = brain_data_factory(config)

        print 'Evaluating...'

        def majority_vote(candidates):
            return np.argmax(np.mean([np.array(c['data']['target']) for c in candidates], axis=0))

        def weighted_majority_vote(candidates):
            votes = [np.exp(-c['dist']) * np.array(c['data']['target']) for c in candidates]
            return np.argmax(np.mean(votes, axis=0))

        brain_database.engine.distance = EuclideanDistance()

        #neighbors = []
        nb_neighbors = 0
        start = time.time()
        nb_success = 0.0
        nb_patches = 0
        for brain_id, (brain, label) in enumerate(brain_data):
            patches_and_pos = get_patches(brain, patch_shape=patch_shape, min_nonempty=args.min_nonempty)
            patches = flattenize((patch for patch, pos in patches_and_pos))

            start_brain = time.time()
            neighbors_per_patch = brain_database.query(patches)
            nb_patches += len(neighbors_per_patch)
            brain_neighbors = list(chain(*neighbors_per_patch))
            print "Brain #{0} ({3:,} patches), found {1:,} neighbors in {2:.2f} sec.".format(brain_id, len(brain_neighbors), time.time()-start_brain, len(neighbors_per_patch))

            #prediction = weighted_majority_vote(brain_neighbors)
            prediction = majority_vote(brain_neighbors)
            nb_success += prediction == np.argmax(label)

            nb_neighbors += len(brain_neighbors)
            del brain_neighbors
            #neighbors.extend(brain_neighbors)

        nb_brains = brain_id + 1
        print "Found a total of {0:,} neighbors for {1} brains ({3:,} patches) in {2:.2f} sec.".format(nb_neighbors, nb_brains, time.time()-start, nb_patches)
        print "Classification error: {:2.2f}%".format(100 * (1. - nb_success/nb_brains))

    elif args.command == "map":
        brain_db = brain_manager[args.name]
        if brain_db is None:
            raise ValueError("Unexisting brain database: " + args.name)

        patch_shape = brain_db.metadata['patch'].shape
        config = json.load(open(args.config))
        brain_data = brain_data_factory(config, pipeline=pipeline)

        total_neg, total_pos = brain_db.labels_count()
        total = float(total_pos + total_neg)
        ratio_neg = (total_neg / total)
        ratio_pos = (total_pos / total)

        print 'Mapping...'
        if args.radius is not None:
            brain_db.engine.distance = EuclideanDistance(brain_db.metadata['patch'])
            brain_db.engine.filters = [NearestFilter(args.k)]
            #brain_db.engine.filters = [DistanceThresholdFilter(10)]

            half_voxel_size = np.array(patch_shape) // 2

            start = time.time()
            for brain_id, brain in enumerate(brain_data):
                patches, positions = brain.extract_patches(patch_shape, min_nonempty=args.min_nonempty, with_positions=True)

                positives = np.zeros_like(brain.image, dtype=np.float32)
                negatives = np.zeros_like(brain.image, dtype=np.float32)

                start_brain = time.time()
                nb_neighbors_per_brain = 0
                nb_empty = 0

                from ipdb import set_trace as dbg
                dbg()

                for patch_id, neighbors in brain_db.get_neighbors_with_pos(patches, positions, args.radius, attributes=["label"]):
                    #patch = patches[patch_id]
                    position = positions[patch_id]
                    neighbors_label = neighbors['label']

                    voxel_pos = tuple(position + half_voxel_size)

                    if len(neighbors_label) <= 0:
                        nb_empty += 1
                        #classif_map[pixel_pos] = 0.5 + OFFSET
                        #parkinson_prob_map[pixel_pos] = 0.5 + OFFSET
                        continue

                    nb_neighbors_per_brain += len(neighbors_label)

                    # Assume binary classification for now
                    negatives[voxel_pos] += np.sum(neighbors_label == 0)
                    positives[voxel_pos] += np.sum(neighbors_label == 1)

                    # pos = nb_pos * ratio_neg
                    # neg = nb_neg * ratio_pos
                    # m = (pos-neg) / len(neighbors_label)
                    # if m > 0:
                    #     m /= ratio_neg
                    # else:
                    #     m /= ratio_pos

                    # classif_map[pixel_pos] = (m+1)/2. + OFFSET
                    # parkinson_prob_map[pixel_pos] = nb_pos/len(neighbors_label) + OFFSET

                print "Brain #{0} ({3:,} patches) found {1:,} neighbors in {2:.2f} sec.".format(brain_id, nb_neighbors_per_brain, time.time()-start_brain, len(patches))
                print "Patches with no neighbors: {:,}".format(nb_empty)

                from ipdb import set_trace as dbg
                dbg()

                RESULTS_FOLDERS = './results_pos'
                if not os.path.isdir(RESULTS_FOLDERS):
                    os.mkdir(RESULTS_FOLDERS)

                def generate_name(prefix, dataset_name, brain_id, dbname, k):
                    if prefix != "":
                        return "{}_{}-{}_db-{}_kNN-{}".format(prefix, dataset_name, brain_id, dbname, k)
                    else:
                        return "{}-{}_db-{}_kNN-{}".format(dataset_name, brain_id, dbname, k)

                name = generate_name(prefix=args.prefix, brain_id=brain_id, dataset_name=brain.name, dbname=brain_db.name, k=args.k)

                parkinson_prob_map = positives / (positives + negatives)
                parkinson_prob_map = np.nan_to_num(parkinson_prob_map)
                parkinson_prob_map[zip(*(positions+half_voxel_size))] += OFFSET

                P = ratio_pos
                N = positives + negatives
                voxel_std = np.sqrt(P*(1-P)/N)
                probs = positives / N
                Z = (probs-P) / voxel_std

                Z *= (positives >= 5).astype('float32')
                Z *= (negatives >= 5).astype('float32')

                parkinson_prob_map = np.nan_to_num(Z)

                save_nifti(brain.image, brain.infos['affine'], pjoin(RESULTS_FOLDERS, "{}_{}.nii.gz".format(brain_data.name, brain.name)))
                #save_nifti(classif_map, brain.infos['affine'], pjoin(RESULTS_FOLDERS, "classif_{}.nii.gz".format(name)))
                save_nifti(parkinson_prob_map, brain.infos['affine'], pjoin(RESULTS_FOLDERS, "parkinson_prob_map_{}.nii.gz".format(name)))

        else:
            brain_db.engine.distance = EuclideanDistance(brain_db.metadata['patch'])
            brain_db.engine.filters = [NearestFilter(args.k)]
            #brain_db.engine.filters = [DistanceThresholdFilter(10)]

            half_patch_size = np.array(patch_shape) // 2

            start = time.time()
            for brain_id, brain in enumerate(brain_data):
                patches, positions = brain.extract_patches(patch_shape, min_nonempty=args.min_nonempty, with_positions=True)

                vectors = patches.reshape((-1, np.prod(patch_shape)))
                if args.use_spatial_code:
                    # Normalize position
                    pos_normalized = positions / np.array(brain.infos['img_shape'], dtype="float32")
                    pos_normalized = pos_normalized.astype("float32")
                    vectors = np.c_[pos_normalized, vectors]

                positives = np.zeros_like(brain.image, dtype=np.float32)
                negatives = np.zeros_like(brain.image, dtype=np.float32)

                start_brain = time.time()
                nb_neighbors_per_brain = 0
                nb_empty = 0

                for patch_id, neighbors in brain_db.get_neighbors(vectors, patches, attributes=["label", "position"]):
                    #patch = patches[patch_id]
                    position = positions[patch_id]
                    neighbors_label = neighbors['label']
                    #neighbors_position = neighbors['position']

                    voxel_pos = tuple(position + half_patch_size)

                    if len(neighbors_label) <= 0:
                        nb_empty += 1
                        continue

                    nb_neighbors_per_brain += len(neighbors_label)
                    negatives[voxel_pos] += np.sum(neighbors_label == 0)
                    positives[voxel_pos] += np.sum(neighbors_label == 1)

                print "Brain #{0} ({3:,} patches) found {1:,} neighbors in {2:.2f} sec.".format(brain_id, nb_neighbors_per_brain, time.time()-start_brain, len(patches))
                print "Patches with no neighbors: {:,}".format(nb_empty)

                from ipdb import set_trace as dbg
                dbg()

                RESULTS_FOLDERS = './results_new'
                if not os.path.isdir(RESULTS_FOLDERS):
                    os.mkdir(RESULTS_FOLDERS)

                def generate_name(prefix, dataset_name, brain_id, dbname, k):
                    if prefix != "":
                        return "{}_{}-{}_db-{}_kNN-{}".format(prefix, dataset_name, brain_id, dbname, k)
                    else:
                        return "{}-{}_db-{}_kNN-{}".format(dataset_name, brain_id, dbname, k)

                name = generate_name(prefix=args.prefix, brain_id=brain_id, dataset_name=brain.name, dbname=brain_db.name, k=args.k)

                metric_map = proportion_test_map(positives, negatives, ratio_pos=ratio_pos)
                save_nifti(brain.image, brain.infos['affine'], pjoin(RESULTS_FOLDERS, "{}_{}.nii.gz".format(brain_data.name, brain.name)))
                save_nifti(metric_map, brain.infos['affine'], pjoin(RESULTS_FOLDERS, "metric_{}.nii.gz".format(name)))

    elif args.command == "vizu":
        from brainsearch.vizu_chaco import NoisyBrainsearchViewer

        brain_db = brain_manager[args.name]
        if brain_db is None:
            raise ValueError("Unexisting brain database: " + args.name)

        patch_shape = brain_db.metadata['patch'].shape
        config = json.load(open(args.config))
        brain_data = brain_data_factory(config, pipeline=pipeline)

        for brain_id, brain in enumerate(brain_data):
            print 'Viewing brain #{0} (label: {1})'.format(brain_id, brain.label)
            patches, positions = brain.extract_patches(patch_shape, min_nonempty=args.min_nonempty, with_positions=True)

            query = {'patches': patches,
                     'positions': positions,
                     'patch_size': patch_shape}
            viewer = NoisyBrainsearchViewer(query, brain_db.engine, brain_voxels=brain.image)
            viewer.configure_traits()

    return brain_manager


if __name__ == '__main__':
    db_manager = main()
    # import sys
    # sys.argv = "brain_search.py --storage memory list".split()
    # db_manager = main(db_manager)

    # sys.argv = "brain_search.py --storage memory -m 3 add SPECTRAL data/FA/PPMI_trainset.json".split()
    # db_manager = main(db_manager)

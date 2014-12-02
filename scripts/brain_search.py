#!/usr/bin/env python
import json
import time
import numpy as np
import pylab as plt
import itertools
import nibabel as nib

from itertools import izip, chain
import brainsearch.vizu as vizu

import brainsearch.utils as brainutil
from brainsearch.imagespeed import blockify
from brainsearch.brain_database import BrainDatabaseManager
from brainsearch.brain_data import brain_data_factory

from nearpy.hashes import RandomBinaryProjections
from nearpy.distances import EuclideanDistance
from nearpy.filters import NearestFilter
from nearpy.utils import chunk, ichunk

import argparse


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


def build_subcommand_init(subparser):
    DESCRIPTION = "Build a new brain database (nearpy's engine)."

    p = subparser.add_parser("init",
                             description=DESCRIPTION,
                             help=DESCRIPTION)

    p.add_argument('name', type=str, help='name of the brain database')
    p.add_argument('shape', metavar="X,Y,...", type=str, help="data's shape or patch shape")
    #p.add_argument('config', type=str, help='contained in a JSON file')
    #p.add_argument('--patch', metavar="X,Y,...", type=str, help='shape of the patches (tuple)')
    #p.add_argument('hashes', metavar="hash", type=str, nargs="+", help='hash functions to use')
    p.add_argument('--LSH', metavar="N", type=int, nargs="+", help='numbers of random projections')


def build_subcommand_add(subparser):
    DESCRIPTION = "Add data to an existing brain database."

    p = subparser.add_parser("add",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('name', type=str, help='name of the brain database')
    p.add_argument('config', type=str, help='contained in a JSON file')
    p.add_argument('-m', dest="min_nonempty", type=int, help='consider only patches having this minimum number of non-empty voxels')
    p.add_argument('--skip', metavar="N", type=int, help='skip N images', default=0)
    p.add_argument('-r', dest="resampling_factor", type=float, help='resample image before processing', default=1.)


def build_subcommand_eval(subparser):
    DESCRIPTION = "Evaluate data given an existing brain database."

    p = subparser.add_parser("eval",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('name', type=str, help='name of the brain database')
    p.add_argument('config', type=str, help='contained in a JSON file')
    p.add_argument('-k', type=int, help='consider at most K nearest-neighbors')
    p.add_argument('-m', dest="min_nonempty", type=int, help='consider only patches having this minimum number of non-empty voxels')


def build_subcommand_map(subparser):
    DESCRIPTION = "Create a color map for a brain given an existing brain database."

    p = subparser.add_parser("map",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('name', type=str, help='name of the brain database')
    p.add_argument('config', type=str, help='contained in a JSON file')
    p.add_argument('-m', dest="min_nonempty", type=int, help='consider only patches having this minimum number of non-empty voxels')
    p.add_argument('-r', dest="resampling_factor", type=float, help='resample image before processing', default=1.)


def build_subcommand_check(subparser):
    DESCRIPTION = "Check candidates distribution given an existing brain database."

    p = subparser.add_parser("check",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('name', type=str, help='name of the brain database')
    p.add_argument('config', type=str, nargs='?', help='contained in a JSON file')
    p.add_argument('-m', dest="min_nonempty", type=int, help='consider only patches having this minimum number of non-empty voxels')


def buildArgsParser():
    DESCRIPTION = "Script to perform brain searches."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    subparser = p.add_subparsers(title="brain_search commands", metavar="", dest="command")
    build_subcommand_list(subparser)
    build_subcommand_init(subparser)
    build_subcommand_add(subparser)
    build_subcommand_eval(subparser)
    build_subcommand_map(subparser)
    build_subcommand_check(subparser)
    build_subcommand_clear(subparser)

    return p


def get_targets(config):
    for source in config["sources"]:
        if source["type"] == "file":
            infos = np.load(source["path"])
            for target in infos["targets"]:
                yield target


def get_data(config):
    for source in config["sources"]:
        if source["type"] == "file":
            infos = np.load(source["path"])
            if 'targets' in infos:
                yield infos["data"], infos["targets"]
            else:
                yield infos["data"], []


def get_patches(brain, patch_shape, min_nonempty=None):
    return blockify(brain, patch_shape, min_nonempty=min_nonempty)
    #return iter(blockify(brain, patch_shape, min_nonempty=min_nonempty))
    #for patch, pos in :
    #    yield patch, pos


def flattenize(iterable):
    for e in iterable:
        yield e.flatten()


def get_patches_with_info(brain, brain_id, label, patch_shape, min_nonempty=None):
    patches, positions = get_patches(brain, patch_shape, min_nonempty=min_nonempty)
    nb_patches = len(patches)
    infos = {"id": np.ones(nb_patches, dtype=np.int32) * brain_id,
             "label": np.ones(nb_patches, dtype=np.int8) * label,
             "position": positions}

    return patches, infos

    # patches, positions = get_patches(brain, patch_shape, min_nonempty=min_nonempty)
    # for chunk_patches, chunk_positions in izip(chunk(patches, n=200000), chunk(positions, n=200000)):
    #     nb_patches = len(chunk_patches)
    #     infos = {"id": np.ones(nb_patches, dtype=np.int32) * brain_id,
    #              "target": np.ones(nb_patches, dtype=np.int8) * label,
    #              "position": chunk_positions}

    #     yield chunk_patches, infos


def save_nifti(image, affine, name):
    nifti = nib.Nifti1Image(image, affine)
    nib.save(nifti, name)


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    brain_manager = BrainDatabaseManager()

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

        if args.name is not None:
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
            brain_manager.remove_all_brain_databases()
        else:
            for name in args.names:
                brain_db = brain_manager[name]
                if brain_db is None:
                    raise ValueError("Unexisting brain database: " + name)

                print "Clearing", name
                brain_manager.remove_brain_database(brain_db)

        print "Done in {:.2f} sec.".format(time.time()-start)

    elif args.command == "init":
        print "Adding", args.name
        shape = tuple(map(int, args.shape.split(",")))

        hashes = []
        for nb_projections in args.LSH:
            hash_name = "LSH{nb_projections}".format(nb_projections=nb_projections)
            hashes.append(RandomBinaryProjections(hash_name, nb_projections, dimension=np.prod(shape)))

        metadata = {"patch": {"dtype": np.dtype(np.float32).str, "shape": shape},
                    "label": {"dtype": np.dtype(np.int8).str, "shape": (1,)},
                    "id": {"dtype": np.dtype(np.int32).str, "shape": (1,)},
                    "position": {"dtype": np.dtype(np.int32).str, "shape": (2,)},
                    }
        brain_manager.new_brain_database(args.name, hashes[0], metadata)

    elif args.command == "add":
        brain_db = brain_manager[args.name]
        if brain_db is None:
            raise ValueError("Unexisting brain database: " + args.name)

        patch_shape = brain_db.metadata['patch'].shape
        config = json.load(open(args.config))
        brain_data = brain_data_factory(config, skip=args.skip)

        print 'Inserting...'
        nb_elements_total = 0
        start = time.time()
        for brain_id, brain in enumerate(brain_data, start=args.skip):
            if brain_id > 30: break
            start_brain = time.time()
            image, affine = brain.resample(args.resampling_factor)
            patches, datainfo = get_patches_with_info(image, brain_id, brain.label, patch_shape=patch_shape, min_nonempty=args.min_nonempty)
            patches -= patches.mean(dtype=np.float64)
            patches /= patches.std(dtype=np.float64)
            #nb_elements_added = brain_database.insert_all(patches, datainfo)
            hashkeys = brain_db.insert(patches, datainfo["label"], datainfo["position"], datainfo["id"])

            # nb_elements_added = 0
            # for patches, datainfo in get_patches_with_info(brain, brain_id, label, patch_shape=patch_shape, min_nonempty=args.min_nonempty):
            #     nb_elements_added += brain_database.insert_all(patches, datainfo)
            #     print "{:,} patches added".format(nb_elements_added)

            print "ID: {0} (label:{3}), {1:,} patches in {2:.2f} sec.".format(brain_id, len(hashkeys), time.time()-start_brain, brain.label)
            nb_elements_total += len(hashkeys)

            if brain_id == 100:
                break

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

            brain_db.show_large_buckets(sizes, bucketkeys)
        else:
            patch_shape = brain_db.metadata['patch'].shape
            config = json.load(open(args.config))
            brain_data = brain_data_factory(config)

            print 'Counting...'
            candidate_count_per_brain = []
            start = time.time()
            for brain_id, (brain, _) in enumerate(brain_data):
                patches, positions = get_patches(brain, patch_shape=patch_shape, min_nonempty=args.min_nonempty)

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
        brain_data = brain_data_factory(config)

        pos_count, neg_count = brain_db.labels_count()

        print 'Mapping...'
        if "mnist" in args.name.lower():
            #brain_database.engine.distance = EuclideanDistance()
            #brain_database.engine.vector_filters = [NearestFilter(100)]

            start = time.time()
            for brain_id, (brain, label) in enumerate(brain_data):
                patches, positions = get_patches(brain, patch_shape=patch_shape, min_nonempty=args.min_nonempty)

                image = np.array([brain.T, brain.T, brain.T]).T
                start_brain = time.time()
                nb_neighbors_per_brain = 0

                neighbors = brain_db.get_neighbors(patches)
                for id_patch, (patch, position, neighbors_label) in enumerate(izip(patches, positions, neighbors['target'])):
                    if len(neighbors_label) == 0:
                        #image[pos[0] + patch_shape[0]//2, pos[1] + patch_shape[1]//2, :] *= [1, 0, 0]
                        continue
                    nb_neighbors_per_brain += len(neighbors_label)

                    neighbors_label = neighbors_label.copy()
                    neighbors_label[neighbors_label == 6] = 0
                    neighbors_label[neighbors_label == 8] = 1

                    # Assume binary classification for now
                    pos = np.mean(neighbors_label)
                    neg = 1 - pos

                    pos_color = np.array([0., 1., 0.])
                    neg_color = np.array([0., 0., 1.])
                    color = pos * pos_color + neg * neg_color

                    image[position[0] + patch_shape[0]//2, position[1] + patch_shape[1]//2, :] *= color

                print "Brain #{0} ({3:,} patches) found {1:,}, neighbors in {2:.2f} sec.".format(brain_id, nb_neighbors_per_brain, time.time()-start_brain, id_patch+1)

                plt.imshow(image, interpolation="nearest")
                #plt.imshow(image)
                plt.show()
        else:
            #brain_database.engine.distance = EuclideanDistance()
            #brain_database.engine.vector_filters = [NearestFilter(100)]

            start = time.time()
            for brain_id, brain in enumerate(brain_data):
                image, affine = brain.resample(args.resampling_factor)
                patches, positions = get_patches(image, patch_shape=patch_shape, min_nonempty=args.min_nonempty)
                patches -= patches.mean(dtype=np.float64)
                patches /= patches.std(dtype=np.float64)

                #classif_map = np.array([brain.T, brain.T, brain.T]).T.copy() / brain.max()
                #classif_map = np.zeros(image.shape + (3,), dtype=np.float32)
                classif_map = np.zeros_like(image, dtype=np.float32)
                start_brain = time.time()
                nb_neighbors_per_brain = 0
                nb_empty = 0

                for chunk_patches, chunk_positions in izip(chunk(patches, n=10000), chunk(positions, n=10000)):
                    neighbors = brain_db.get_neighbors(chunk_patches, attributes=["label"])
                    for id_patch, (patch, position, neighbors_label) in enumerate(izip(chunk_patches, chunk_positions, neighbors['label'])):
                        if len(neighbors_label) == 0:
                            nb_empty += 1
                            continue

                        nb_neighbors_per_brain += len(neighbors_label)

                        # Assume binary classification for now
                        nb_pos = np.sum(neighbors_label, dtype=np.float64)
                        nb_neg = len(neighbors_label) - nb_pos
                        pos = nb_pos / pos_count
                        neg = nb_neg / neg_count
                        sum_pos_neg = pos + neg
                        pos /= sum_pos_neg
                        #neg /= sum_pos_neg

                        classif_map[position[0] + patch_shape[0]//2, position[1] + patch_shape[1]//2, position[2] + patch_shape[2]//2] = pos

                        #pos_color = np.array([0., 1., 0.], dtype=np.float32)
                        #neg_color = np.array([0., 0., 1.], dtype=np.float32)
                        #color = pos * pos_color + neg * neg_color
                        #classif_map[position[0] + patch_shape[0]//2, position[1] + patch_shape[1]//2, position[2] + patch_shape[2]//2, :] = color[:]

                print "Brain #{0} ({3:,} patches) found {1:,} neighbors in {2:.2f} sec.".format(brain_id, nb_neighbors_per_brain, time.time()-start_brain, len(patches))
                print "Patches with no neighbors: {:,}".format(nb_empty)
                #plt.imshow(classif_map[:, 25, :, :], interpolation="nearest")
                #plt.imshow(classif_map[:, 25, :], interpolation="nearest")
                #plt.show()

                save_nifti(image, affine, "T1_{}.nii.gz".format(brain_id))
                save_nifti(classif_map, affine, "classif_{}.nii.gz".format(brain_id))


if __name__ == '__main__':
    main()





# def get_data_for_query(config, patch_shape=None):
#     for source in config["sources"]:
#         if source["type"] == "file":
#             infos = np.load(source["path"])
#             if patch_shape is not None and tuple(source["shape"]) != patch_shape:
#                 for idx, (datum, target) in enumerate(izip(infos["data"], infos["targets"])):
#                     for block, pos in blockify(datum, patch_shape, keep_empty=False):
#                         yield block.flatten()
#             else:
#                 for idx, (datum, target) in enumerate(izip(infos["data"], infos["targets"])):
#                     yield datum.flatten()


# def get_data(config, patch_shape=None):
#     for source in config["sources"]:
#         if source["type"] == "file":
#             infos = np.load(source["path"])
#             if patch_shape is not None and tuple(source["shape"]) != patch_shape:
#                 for idx, (datum, target) in enumerate(izip(infos["data"], infos["targets"])):
#                     for block, pos in blockify(datum, patch_shape, keep_empty=False):
#                         datum_info = {"id": idx,
#                                       "target": target.tolist(),
#                                       "position": pos.tolist()}
#                         yield block.flatten(), datum_info
#             else:
#                 for idx, (datum, target) in enumerate(itertools.izip(infos["data"], infos["targets"])):
#                     datum_info = {"id": idx,
#                                   "target": target.tolist(),
#                                   "position": (0, 0)}
#                     yield datum.flatten(), datum_info
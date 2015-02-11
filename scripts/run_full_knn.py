#!/usr/bin/env python

import json
import argparse
import numpy as np
import nibabel as nib
from time import time

from brainsearch.imagespeed import blockify
from brainsearch.brain_data import brain_data_factory

import theano
import theano.tensor as T


def find_kNN2(query_patch, k, dataset, patch_shape, min_nonempty):
    config = json.load(open(dataset))
    brain_data = brain_data_factory(config)

    best_distances = np.empty((0, 1), dtype="float32")
    best_labels = np.empty((0, 1), dtype="int8")
    best_ids = np.empty((0, 1), dtype="int32")
    best_positions = np.empty((0, 3), dtype="int32")

    for brain_id, brain in enumerate(brain_data):
        patches, datainfo = brain.extract_patches(patch_shape, min_nonempty=min_nonempty, with_info=True)
        distances = np.sum((patches - query_patch)**2, axis=(1, 2, 3))
        top_k = np.argsort(distances)[:k]

        distances = distances[top_k]
        labels = datainfo["label"][top_k]
        ids = datainfo["id"][top_k]
        positions = datainfo["position"][top_k]

        # Keep best
        best_distances = np.r_[best_distances, distances]
        best_labels = np.r_[best_labels, labels]
        best_ids = np.r_[best_ids, ids]
        best_positions = np.r_[best_positions, positions]

        best_top_k = np.argsort(best_distances)[:k]
        best_distances = best_distances[best_top_k]
        best_labels = best_labels[best_top_k]
        best_ids = best_ids[best_top_k]
        best_positions = best_positions[best_top_k]

    return best_distances, best_labels, best_ids, best_positions


def find_kNN_theano(query_patches, best, k, brain, patch_shape, min_nonempty):
    patches, datainfo = brain.extract_patches(patch_shape, min_nonempty=min_nonempty, with_info=True)

    dataset = theano.shared(patches, name="dataset", borrow=True)
    query = T.tensor3('query')
    distances = T.sum((dataset-query)**2, axis=(1, 2, 3))
    top_k = T.argsort(distances)[:k]

    dist = theano.function([query], [distances, top_k])

    for i, query_patch in enumerate(query_patches):
        if i % 10000 == 0:
            print "{}/{}".format(i, len(query_patches))

        distances, top_k = dist(query_patch)

        distances = distances[top_k]
        labels = datainfo["label"][top_k]
        ids = datainfo["id"][top_k]
        positions = datainfo["position"][top_k]

        # Keep best
        best_distances = np.r_[best[0][i], distances]
        best_labels = np.r_[best[1][i], labels]
        best_ids = np.r_[best[2][i], ids]
        best_positions = np.r_[best[3][i], positions]

        best_top_k = np.argsort(best_distances)[:k]
        best[0][i] = best_distances[best_top_k]
        best[1][i] = best_labels[best_top_k]
        best[2][i] = best_ids[best_top_k]
        best[3][i] = best_positions[best_top_k]


def find_kNN(query_patches, best, k, brain, patch_shape, min_nonempty):
    patches, datainfo = brain.extract_patches(patch_shape, min_nonempty=min_nonempty, with_info=True)

    for i, query_patch in enumerate(query_patches):
        if i % 1 == 0:
            print "\n{}/{}".format(i, len(query_patches))

        #import IPython
        #namespace = locals().copy()
        #namespace.update(globals())
        #IPython.embed(user_ns=namespace, banner1="")

        start = time()
        distances = np.sum((patches - query_patch)**2, axis=(1, 2, 3))
        print time()-start

        start = time()
        top_k = np.argsort(distances)[:k]
        print time()-start

        start = time()
        distances = distances[top_k]
        labels = datainfo["label"][top_k]
        ids = datainfo["id"][top_k]
        positions = datainfo["position"][top_k]

        # Keep best
        best_distances = np.r_[best[0][i], distances]
        best_labels = np.r_[best[1][i], labels]
        best_ids = np.r_[best[2][i], ids]
        best_positions = np.r_[best[3][i], positions]

        best_top_k = np.argsort(best_distances)[:k]
        best[0][i] = best_distances[best_top_k]
        best[1][i] = best_labels[best_top_k]
        best[2][i] = best_ids[best_top_k]
        best[3][i] = best_positions[best_top_k]
        print time()-start


def buildArgsParser():
    p = argparse.ArgumentParser(description="Perform a full kNN")

    p.add_argument('query', type=str, help='nifti file')
    p.add_argument('dataset', type=str, help='dataset in a JSON file')
    p.add_argument('--shape', metavar="X,Y,...", type=str, help="patch shape", default=(5, 5, 5))
    p.add_argument('-k', dest="k", type=int, help='numbers of nearest neighbors', default=100)
    p.add_argument('-m', dest="min_nonempty", type=int, help='consider only patches having this minimum number of non-empty voxels', default=1)

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    nii = nib.load(args.query)
    image = nii.get_data()
    patches, positions = blockify(image, args.shape, min_nonempty=args.min_nonempty)

    best_distances = np.empty((len(patches), args.k), dtype="float32")
    best_labels = np.empty((len(patches), args.k), dtype="int8")
    best_ids = np.empty((len(patches), args.k), dtype="int32")
    best_positions = np.empty((len(patches), args.k, 3), dtype="int32")
    best = best_distances, best_labels, best_ids, best_positions

    brain_data = brain_data_factory(json.load(open(args.dataset)))

    for brain in brain_data:
        start = time()
        find_kNN(patches, best, args.k, brain, args.shape, args.min_nonempty)
        #find_kNN_theano(patches, best, args.k, brain, args.shape, args.min_nonempty)
        print "Brain #{}, done in {:.2f} sec".format(brain.id, time()-start)

        from ipdb import set_trace as dbg
        dbg()

    from ipdb import set_trace as dbg
    dbg()

    import pickle
    pickle.dump(best, open('full_knn.pkl', 'w'))

if __name__ == '__main__':
    main()

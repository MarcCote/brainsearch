#!/usr/bin/env python
from __future__ import division

import json
import argparse
import numpy as np
import nibabel as nib
from time import time

from brainsearch.imagespeed import blockify
from brainsearch.brain_data import brain_data_factory

import theano
import theano.tensor as T
from os.path import join as pjoin


# Still too slow on GPU
def find_kNN_theano(query_patches, best, k, brain, patch_shape, min_nonempty):
    patches, datainfo = brain.extract_patches(patch_shape, min_nonempty=min_nonempty, with_info=True)

    BATCH_SIZE = 5
    dataset = theano.shared(patches, name="dataset", borrow=True)[:, None, :, :, :]
    query = T.tensor4('query')
    distances = T.sum((dataset-query)**2, axis=(2, 3, 4))
    top_k = T.argsort(distances, axis=0)[:k]

    #dist = theano.function([query], [distances[top_k, range(BATCH_SIZE)], top_k])
    dist = theano.function([query], distances)

    #for i, query_patch in enumerate(query_patches):
    for i in range(0, len(query_patches), BATCH_SIZE):
        #if i % 10000 == 0:
        print "{}/{}".format(i, len(query_patches))

        start = time()
        distances = dist(query_patches[i:i+BATCH_SIZE])
        top_k = np.min(distances, axis=0)
        print time()-start

        from ipdb import set_trace
        set_trace()

        """
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
        """


def find_kNN(query_patches, best, k, brain, patch_shape, min_nonempty):
    patches, datainfo = brain.extract_patches(patch_shape, min_nonempty=min_nonempty, with_info=True)

    for i, query_patch in enumerate(query_patches):
        distances = np.sum((patches - query_patch)**2, axis=(1, 2, 3))
        top_k = np.argsort(distances)[:k]

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


def buildArgsParser():
    p = argparse.ArgumentParser(description="Perform a full kNN")

    p.add_argument('query', type=str, help='nifti file')
    p.add_argument('dataset', type=str, help='dataset in a JSON file')
    p.add_argument('--shape', metavar="X,Y,...", type=str, help="patch shape", default=(5, 5, 5))
    p.add_argument('-k', dest="k", type=int, help='numbers of nearest neighbors', default=100)
    p.add_argument('-m', dest="min_nonempty", type=int, help='consider only patches having this minimum number of non-empty voxels', default=1)

    p.add_argument('--brain_id', type=int, help='ID of the brain in the dataset.')
    p.add_argument('--batch_id', type=int, help='ID of the batch of query patches to be processed.')
    p.add_argument('--batch_size', type=int, help='Number of query patches in a batch.', default=1)
    p.add_argument('--out', type=str, help='Directory where to save intermediate results.', default='./')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    nii = nib.load(args.query)
    image = nii.get_data()
    patches, positions = blockify(image, args.shape, min_nonempty=args.min_nonempty)

    if args.batch_id is None:
        start = args.batch_id * args.batch_size
        end = (args.batch_id+1) * args.batch_size
        patches = patches[start:end]

    best_distances = np.empty((len(patches), args.k), dtype="float32")
    best_labels = np.empty((len(patches), args.k), dtype="int8")
    best_ids = np.empty((len(patches), args.k), dtype="int32")
    best_positions = np.empty((len(patches), args.k, 3), dtype="int32")
    best = best_distances, best_labels, best_ids, best_positions

    brain_data = brain_data_factory(json.load(open(args.dataset)))

    for brain in brain_data:
        if args.brain_id != brain.id:
            continue

        print "Processing brain #{} ...".format(brain.id)
        start = time()
        find_kNN(patches, best, args.k, brain, args.shape, args.min_nonempty)
        print "Brain #{}, done in {:.2f} sec".format(brain.id, time()-start)

    name = "{batch_id}_{brain_id}.npz".format(batch_id=args.batch_id, brain_id=args.brain_id)
    np.savez(pjoin(args.out, name), dist=best_distances, labels=best_labels, ids=best_ids, positions=best_positions)

    import pickle
    pickle.dump(best, open('full_knn.pkl', 'w'))

if __name__ == '__main__':
    main()

import os
import nibabel as nib
import numpy as np
from itertools import izip_longest

from brainsearch.imagespeed import blockify
from brainsearch.brain_processing import BrainPipelineProcessing


def brain_data_factory(config, pipeline=BrainPipelineProcessing(), id=None):
    name = config["name"]
    sources = config["sources"]
    if config["type"] == "numpy":
        return NumpyBrainData(name=name, sources=sources, pipeline=pipeline, id=id)
    elif config["type"] == "nifti":
        return NiftiBrainData(name=name, sources=sources, pipeline=pipeline, id=id)


class BrainPatches(object):
    def __init__(self, brain, patches, positions):
        self.brain = brain
        self.patches = patches
        self.positions = positions
        self._brain_ids = None
        self._labels = None

    def __len__(self):
        return len(self.patches)

    @property
    def brain_ids(self):
        if self._brain_ids is None:
            self._brain_ids = np.ones(len(self), dtype=np.int32) * self.brain.id

        return self._brain_ids

    @property
    def labels(self):
        if self._labels is None:
            self._labels = np.ones(len(self), dtype=np.int8) * self.brain.label

        return self._labels

    def create_vectors(self, spatial_weight=0.):
        vectors = self.patches.reshape((len(self), -1))

        if spatial_weight > 0.:
            # Normalize position
            pos_normalized = self.positions / np.array(self.brain.infos['img_shape'], dtype="float32")
            pos_normalized = spatial_weight*pos_normalized.astype("float32")
            vectors = np.c_[pos_normalized, vectors]

        return vectors


class Brain(object):
    def __init__(self, image, id, name, label, mask=None, **infos):
        self.image = image
        self.id = id
        self.name = name
        self.label = label
        self.infos = infos
        self.mask = mask

    def extract_patches(self, patch_shape, min_nonempty=None):
        if min_nonempty > np.prod(patch_shape):
            raise ValueError("min_nonempty must be smaller than nb. of voxels in a patch!")

        patches, positions = blockify(self.image, patch_shape, min_nonempty_ratio=min_nonempty)

        if self.mask is not None:
            half_patch_size = np.array(patch_shape) // 2
            center_positions = positions + half_patch_size
            indices = []
            for pos in zip(*np.where(self.mask)):
                idx = np.where(np.all(center_positions == pos, axis=1))[0][0]
                indices.append(idx)

            patches = patches[indices]
            positions = positions[indices]

        return BrainPatches(self, patches, positions)


class BrainData(object):
    def __init__(self, name, sources, pipeline=BrainPipelineProcessing(), id=None):
        self.name = name
        self.sources = sources
        self.id = id
        self.pipeline = pipeline

    def __len__(self):
        return len(self.sources)

    def __iter__(self):
        raise NotImplementedError


class NiftiBrainData(BrainData):
    def __init__(self, *args, **kwargs):
        super(NiftiBrainData, self).__init__(*args, **kwargs)

    def __iter__(self):
        for i, source in enumerate(self.sources):
            try:
                if self.id is not None and i != self.id:
                    continue

                name = source['name'] if 'name' in source else os.path.basename(source['path']).split(".nii")[0]
                id = source['id'] if 'id' in source else i

                label = source['label']
                img = nib.load(source['path'])
                brain = img.get_data()

                mask = None
                if "mask" in source:
                    nii_mask = nib.load(source['mask'])
                    mask = nii_mask.get_data().astype(bool)

                brain = Brain(image=np.asarray(brain, dtype=np.float32), id=id, name=name, label=np.int8(label),
                              mask=mask,
                              affine=img.get_affine(), pixeldim=img.get_header().get_zooms()[:3],
                              img_shape=img.shape)

                self.pipeline.process(brain)
                yield brain
            except IOError:
                print "Cannot find {}. Skipping it.".format(source['path'])


class NumpyBrainData(BrainData):
    def __iter__(self):
        for source in self.sources:
            arrays = np.load(source['path'])
            brains = arrays[source['brain']]
            labels = []
            if source['label'] is not None:
                labels = arrays[source['label']]

            for brain, label in izip_longest(brains, labels):
                yield np.asarray(brain, dtype=np.float32), np.int8(np.where(label)[0][0])

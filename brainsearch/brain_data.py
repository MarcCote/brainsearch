import nibabel as nib
import numpy as np
from itertools import izip_longest

from brainsearch.imagespeed import blockify
from brainsearch.brain_processing import BrainPipelineProcessing


def brain_data_factory(config, skip=0, pipeline=BrainPipelineProcessing()):
    name = config["name"]
    sources = config["sources"]
    if config["type"] == "numpy":
        return NumpyBrainData(name=name, sources=sources, skip=skip, pipeline=pipeline)
    elif config["type"] == "nifti":
        return NiftiBrainData(name=name, sources=sources, skip=skip, pipeline=pipeline)


class Brain(object):
    def __init__(self, image, id, label, **infos):
        self.image = image
        self.id = id
        self.label = label
        self.infos = infos

    def extract_patches(self, patch_shape, min_nonempty=None, with_info=False, with_positions=False):
        patches, positions = blockify(self.image, patch_shape, min_nonempty=min_nonempty)

        if not with_info:
            if with_positions:
                return patches, positions

            return patches

        nb_patches = len(patches)
        infos = {"id": np.ones(nb_patches, dtype=np.int32) * self.id,
                 "label": np.ones(nb_patches, dtype=np.int8) * self.label,
                 "position": positions}

        return patches, infos


class BrainData(object):
    def __init__(self, name, sources, skip=0, pipeline=BrainPipelineProcessing()):
        self.name = name
        self.sources = sources
        self.skip = skip
        self.pipeline = pipeline

    def __len__(self):
        return len(self.sources) - self.skip

    def __iter__(self):
        raise NotImplementedError


class NiftiBrainData(BrainData):
    def __init__(self, *args, **kwargs):
        super(NiftiBrainData, self).__init__(*args, **kwargs)

    def __iter__(self):
        for i, source in enumerate(self.sources):
            if i < self.skip:
                continue

            label = source['label']
            img = nib.load(source['path'])
            brain = img.get_data()

            brain = Brain(image=np.asarray(brain, dtype=np.float32), id=i, label=np.int8(label),
                          affine=img.get_affine(), pixeldim=img.get_header().get_zooms()[:3])

            self.pipeline.process(brain)
            yield brain


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
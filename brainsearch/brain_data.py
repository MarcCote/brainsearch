import nibabel as nib
import numpy as np
from itertools import izip_longest

from dipy.align.aniso2iso import resample


def brain_data_factory(config, skip=0):
    name = config["name"]
    sources = config["sources"]
    if config["type"] == "numpy":
        return NumpyBrainData(name=name, sources=sources, skip=skip)
    elif config["type"] == "nifti":
        return NiftiBrainData(name=name, sources=sources, skip=skip)


class Brain(object):
    def __init__(self, image, label, affine, pixeldim):
        self.image = image
        self.label = label
        self.affine = affine
        self.pixeldim = pixeldim

    def resample(self, resampling_factor, order=1):
        #orders = {'nn': 0, 'lin': 1, 'quad': 2, 'cubic': 3}
        new_pixeldim = tuple(resampling_factor * np.asarray(self.pixeldim))
        rimage, raffine = resample(self.image, self.affine, self.pixeldim, new_pixeldim, order=order)
        return rimage, raffine


class BrainData(object):
    def __init__(self, name, sources, skip=0):
        self.name = name
        self.sources = sources
        self.skip = skip

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

            yield Brain(image=np.asarray(brain, dtype=np.float32), label=np.int8(label),
                        affine=img.get_affine(), pixeldim=img.get_header().get_zooms()[:3])


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

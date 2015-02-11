import numpy as np
from dipy.align.aniso2iso import resample
from skimage import exposure


class BrainProcessing(object):
    def process(self, image):
        raise NotImplemented()


class BrainPipelineProcessing(object):
    def __init__(self):
        self.processings = []

    def add(self, processing):
        """
        Parameters
        ----------
        processing : `BrainProcessing` object
        """
        self.processings.append(processing)

    def process(self, brain):
        for processing in self.processings:
            processing.process(brain)


class BrainResampling(BrainProcessing):
    def __init__(self, factor, order=1):
        """
        Parameters
        ----------
        orders : {'nn': 0, 'lin': 1, 'quad': 2, 'cubic': 3}
        """
        self.factor = factor
        self.order = order

    def process(self, brain):
        new_pixeldim = tuple(self.factor * np.asarray(brain.infos['pixeldim']))

        brain.image, brain.infos['affine'] = resample(brain.image, brain.infos['affine'],
                                                      brain.infos['pixeldim'], new_pixeldim,
                                                      order=self.order)


class BrainNormalization(BrainProcessing):
    def __init__(self, type):
        """
        Parameters
        ----------
        type : {'hist_eq': 0, 'minmax': 1, 'zscores': 2}
        """
        self.type = type

    def process(self, brain):
        indices = np.where(brain.image)
        if self.type == 0:  # hist_equalization
            brain.image[indices] = exposure.equalize_hist(brain.image[indices]/np.max(brain.image[indices])).astype(np.float32)
        elif self.type == 1:  # minmax_normalization
            brain.image[indices] -= np.min(brain.image[indices])
            brain.image[indices] /= np.max(brain.image[indices])
        elif self.type == 2:  # zscore_normalization
            brain.image[indices] -= np.mean(brain.image[indices], dtype=np.float64)
            brain.image[indices] /= np.std(brain.image[indices], dtype=np.float64)

import operator
import itertools

from nearpy.utils import Timer


def split(iterable, n=2):
    tees = itertools.tee(iterable, n)
    iterables = []
    for i, tee in enumerate(tees):
        iterables.append(itertools.imap(operator.itemgetter(i), tee))

    return tuple(iterables)


class SmartGenerator(object):
    def __init__(self, genfct, keep_copy=False):
        self.genfct = genfct
        self.keep_copy = keep_copy
        self.elements = None

    def __iter__(self):
        if self.keep_copy:
            if self.elements is not None:
                return iter(self.elements)
            else:
                self.elements = []

                def gen(iterable):
                    for element in iterable:
                        self.elements.append(element)
                        yield element

                return gen(self.genfct())
        else:
            return self.genfct()

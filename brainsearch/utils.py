from __future__ import print_function
import sys
import operator
import itertools
from time import time
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


class Timer2():
    _levels = [0]

    def __init__(self, txt):
        self.txt = txt
        self.duration = 0.
        self.indent = "  " * self._levels[0]
        self._levels[0] += 1

    def __enter__(self):
        self.start = time()
        print(self.indent + self.txt + "... ", end="")
        sys.stdout.flush()

    def __exit__(self, type, value, tb):
        self.duration = time()-self.start
        self._levels[0] -= 1
        print(self.indent + "{:.2f} sec.".format(self.duration))

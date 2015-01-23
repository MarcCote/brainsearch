# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

cdef extern from "imagespeed.h":
    enum: MAX_NDIM

import itertools

import numpy as np
cimport numpy as np

ctypedef int[:] Shape

ctypedef float[:] Data1D
ctypedef float[:,:] Data2D
ctypedef float[:,:,:] Data3D
ctypedef float[:,:,:,:] Data4D
ctypedef float[:,:,:,:,:] Data5D
ctypedef float[:,:,:,:,:,:] Data6D
ctypedef float[:,:,:,:,:,:,:] Data7D

DEF biggest_double = 1.7976931348623157e+308  # np.finfo('f8').max
DEF biggest_int = 2147483647  # np.iinfo('i4').max

ctypedef fused Data:
    Data1D
    Data2D
    Data3D
    Data4D
    Data5D
    Data6D
    Data7D


cdef void _blockify2D(Data2D arr, Shape shape, Data3D out, int[:,:] pos) nogil:
    cdef int x, y, z, i, j, k
    cdef int n = 0

    for y in range(arr.shape[0] - shape[0] + 1):
        for x in range(arr.shape[1] - shape[1] + 1):

            for i in range(shape[0]):
                for j in range(shape[1]):

                    out[n,i,j] = arr[y+i, x+j]
                    pos[n,0] = y
                    pos[n,1] = x

            n += 1


cdef void _blockify3D(Data3D arr, Shape shape, Data4D out, int[:,:] pos) nogil:
    cdef int x, y, z, i, j, k
    cdef int n = 0

    for z in range(arr.shape[0] - shape[0] + 1):
        for y in range(arr.shape[1] - shape[1] + 1):
            for x in range(arr.shape[2] - shape[2] + 1):

                for i in range(shape[0]):
                    for j in range(shape[1]):
                        for k in range(shape[2]):

                            out[n,i,j,k] = arr[z+i, y+j, x+k]
                            pos[n,0] = z
                            pos[n,1] = y
                            pos[n,2] = x

                n += 1


cdef int _blockify2D_nonempty(Data2D arr, Shape shape, int min_nonempty, Data3D out, int[:,:] pos) nogil:
    cdef int x, y, z, i, j, k
    cdef int n = 0
    cdef int nb_empty = 0

    for y in range(arr.shape[0] - shape[0] + 1):
        for x in range(arr.shape[1] - shape[1] + 1):
            nb_empty = 0
            for i in range(shape[0]):
                for j in range(shape[1]):
                    nb_empty += (arr[y+i, x+j] != 0.0)

                    out[n,i,j] = arr[y+i, x+j]
                    pos[n,0] = y
                    pos[n,1] = x

            if nb_empty > min_nonempty:
                n += 1

    return n


cdef int _blockify3D_nonempty(Data3D arr, Shape shape, int min_nonempty, Data4D out, int[:,:] pos) nogil:
    cdef int x, y, z, i, j, k
    cdef int n = 0
    cdef int nb_empty = 0

    for z in range(arr.shape[0] - shape[0] + 1):
        for y in range(arr.shape[1] - shape[1] + 1):
            for x in range(arr.shape[2] - shape[2] + 1):
                nb_empty = 0

                for i in range(shape[0]):
                    for j in range(shape[1]):
                        for k in range(shape[2]):
                            nb_empty += (arr[z+i, y+j, x+k] != 0.0)

                            out[n,i,j,k] = arr[z+i, y+j, x+k]
                            pos[n,0] = z
                            pos[n,1] = y
                            pos[n,2] = x

                if nb_empty > min_nonempty:
                    n += 1

    return n


def blockify(arr, shape, min_nonempty=None):
    """ Split a ndarray `arr` into overlapping blocks of size `shape`.

    Parameters
    ----------
    arr : 2d or 3d array
        Array to split in blocks.
    shape : tuple
        Shape of the block to extract.
    min_nonempty : int, optional
        Tells
    """
    if min_nonempty is None:
        block_shape = np.asarray(shape, dtype=np.int32)
        nb_blocks = np.prod(np.array(arr.shape) - block_shape + 1)
        blocks = np.empty((nb_blocks,) + shape, dtype=np.float32)
        pos = np.empty((nb_blocks, arr.ndim), dtype=np.int32)

        if arr.ndim == 2:
            _blockify2D(arr, block_shape, blocks, pos)
        elif arr.ndim == 3:
            _blockify3D(arr, block_shape, blocks, pos)
        else:
            raise ValueError("Not supported! Only 2D and 3D.")

    else:
        block_shape = np.asarray(shape, dtype=np.int32)
        nb_blocks_max = np.prod(np.array(arr.shape) - block_shape + 1)
        blocks = np.empty((nb_blocks_max,) + shape, dtype=np.float32)
        pos = np.empty((nb_blocks_max, arr.ndim), dtype=np.int32)

        if arr.ndim == 2:
            nb_blocks = _blockify2D_nonempty(arr, block_shape, min_nonempty, blocks, pos)
        elif arr.ndim == 3:
            nb_blocks = _blockify3D_nonempty(arr, block_shape, min_nonempty, blocks, pos)
        else:
            raise ValueError("Not supported! Only 2D and 3D.")

        blocks.resize((nb_blocks,) + shape)
        pos.resize((nb_blocks, arr.ndim))

    #return itertools.izip(blocks, pos)
    return blocks, pos

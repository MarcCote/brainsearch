import numpy as np
from brainsearch.imagespeed import blockify

from nose.tools import assert_equal


def test_blockify():
    # Test 2D
    shape = (6, 6)
    block_shape = (6, 6)
    data = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    results = list(blockify(data, block_shape, min_nonempty=True))

    assert_equal(len(results), 1)

    block_shape = (3, 3)
    nb_blocks = np.prod(np.array(data.shape) - np.array(block_shape) + 1)
    results = list(blockify(data, block_shape, min_nonempty=True))
    assert_equal(len(results), nb_blocks)

    block_shape = (2, 3)
    nb_blocks = np.prod(np.array(data.shape) - np.array(block_shape) + 1)
    results = list(blockify(data, block_shape, min_nonempty=True))
    assert_equal(len(results), nb_blocks)

    # Test 3D
    shape = (6, 6, 6)
    block_shape = (6, 6, 6)
    data = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    results = list(blockify(data, block_shape, min_nonempty=True))

    assert_equal(len(results), 1)

    block_shape = (3, 3, 3)
    nb_blocks = np.prod(np.array(data.shape) - np.array(block_shape) + 1)
    results = list(blockify(data, block_shape, min_nonempty=True))
    assert_equal(len(results), nb_blocks)

    block_shape = (1, 2, 3)
    nb_blocks = np.prod(np.array(data.shape) - np.array(block_shape) + 1)
    results = list(blockify(data, block_shape, min_nonempty=True))
    assert_equal(len(results), nb_blocks)


def test_blockify_nonempty():
    # Test 2D
    shape = (6, 6)
    block_shape = (6, 6)
    data = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    results = list(blockify(data, block_shape, min_nonempty=1))

    assert_equal(len(results), 1)

    block_shape = (3, 3)
    nb_blocks = np.prod(np.array(data.shape) - np.array(block_shape) + 1)
    results = list(blockify(data, block_shape, min_nonempty=1))
    assert_equal(len(results), nb_blocks)

    block_shape = (2, 3)
    nb_blocks = np.prod(np.array(data.shape) - np.array(block_shape) + 1)
    results = list(blockify(data, block_shape, min_nonempty=1))
    assert_equal(len(results), nb_blocks)

    # Put an empty block
    data[:block_shape[0], :block_shape[1]] = 0.0
    results = list(blockify(data, block_shape, min_nonempty=1))
    assert_equal(len(results), nb_blocks-1)

    # Test 3D
    shape = (6, 6, 6)
    block_shape = (6, 6, 6)
    data = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    results = list(blockify(data, block_shape, min_nonempty=1))

    assert_equal(len(results), 1)

    block_shape = (3, 3, 3)
    nb_blocks = np.prod(np.array(data.shape) - np.array(block_shape) + 1)
    results = list(blockify(data, block_shape, min_nonempty=1))
    assert_equal(len(results), nb_blocks)

    block_shape = (1, 2, 3)
    nb_blocks = np.prod(np.array(data.shape) - np.array(block_shape) + 1)
    results = list(blockify(data, block_shape, min_nonempty=1))
    assert_equal(len(results), nb_blocks)

    # Put an empty block
    data[:block_shape[0], :block_shape[1], :block_shape[2]] = 0.0
    results = list(blockify(data, block_shape, min_nonempty=1))
    assert_equal(len(results), nb_blocks-1)

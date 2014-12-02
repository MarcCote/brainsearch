import json
import pickle

import numpy as np
from redis import Redis
from nearpy import Engine
from nearpy.storage import RedisStorage
from nearpy.storage import CRedisStorage

from nearpy.filters import NearestFilter
from nearpy.data import NumpyData

from collections import defaultdict


class BrainDatabase(object):
    def __init__(self, name, redis, engine):
        self.name = name
        self.redis = redis
        self.engine = engine

        #Initialize metadata
        metadata = defaultdict(lambda: {})
        metadata_key = self.name + "_metadata"
        for key, value in self.redis.hscan_iter(metadata_key):
            attribute_name, attribute_info = key.split('_')
            if attribute_info == "shape":
                metadata[attribute_name][attribute_info] = eval(value)
            elif attribute_info == "dtype":
                metadata[attribute_name][attribute_info] = np.dtype(value)
            else:
                metadata[attribute_name][attribute_info] = value

        self._metadata = {}
        for key, value in metadata.items():
            self._metadata[key] = NumpyData(key, value['dtype'], value['shape'])

    @property
    def metadata(self):
        return self._metadata

    def nb_patches(self, check_integrity=False):
        nb_patches = self.redis.hget(self.name, "nb_patches")
        nb_patches = int(nb_patches) if nb_patches is not None else 0

        if check_integrity:
            true_nb_patches = self.engine.nb_patches()
            if true_nb_patches != nb_patches:
                self.update(nb_patches=true_nb_patches-nb_patches)
                nb_patches = true_nb_patches

        return nb_patches

    def nb_buckets(self):
        return self.engine.nb_buckets()

    def buckets_size(self):
        return self.engine.buckets_size()

    def show_large_buckets(self, sizes, bucketkeys):
        from brainsearch import vizu
        indices = np.argsort(sizes)[::-1]

        for idx in indices:
            print "{:,} neighbors".format(sizes[idx])
            patches = self.engine.storage.retrieve_by_key(bucketkeys[idx], attribute=self.metadata['patch'])
            vizu.show_images3d(patches, shape=self.metadata['patch'].shape, blocking=True)

    def labels_count(self, check_integrity=False):
        labels_count = list(self.redis.hscan_iter(self.name, match="label_count_*", count=2))

        if len(labels_count) > 0:
            labels_count = map(int, zip(*labels_count)[1])

        if check_integrity:
            true_labels_count = self.engine.targets_count()
            if np.any(true_labels_count != labels_count):
                updated_labels_count = true_labels_count.copy()
                updated_labels_count[:len(labels_count)] -= labels_count
                self.update(labels_count=updated_labels_count)
                labels_count = true_labels_count

        return labels_count

    def set_metadata(self, infos):
        for attribute, metadata in infos.items():
            self.engine.set_metadata(attribute, metadata)

    def insert(self, patches, labels, positions, brain_ids):
        data = {}
        data[self.metadata['label']] = labels
        data[self.metadata['position']] = positions
        data[self.metadata['id']] = brain_ids

        hashkeys = self.engine.store_batch(patches, data)
        self.update(nb_patches=len(patches))
        self.update(labels_count=np.bincount(labels))
        return hashkeys

    def get_neighbors(self, patches, attributes=None):
        if attributes is None:
            attributes = ['patch', 'label', 'position', 'id']

        for i, attribute in enumerate(attributes):
            attributes[i] = self.metadata[attribute]

        return self.engine.neighbors_batch(patches, *attributes)

    def update(self, nb_patches=None, labels_count=None):
        if nb_patches is not None:
            self.redis.hincrby(self.name, "nb_patches", nb_patches)

        if labels_count is not None:
            for i, count in enumerate(labels_count):
                self.redis.hincrby(self.name, "label_count_{0}".format(i), count)

    def candidate_count(self, patches):
        candidate_count = self.engine.candidate_count_batch(patches)
        return candidate_count

    def query(self, data, k=None):
        if k is not None:
            self.engine.vector_filters = [NearestFilter(k)]

        return self.engine.neighbors_from_iter(data)


class BrainDatabaseManager(object):
    DATABASES_LIST_KEY = "BRAIN_DB"

    def __init__(self, host='localhost', port=6379):
        self.redis = Redis(host=host, port=port, db=0)
        self.brain_databases = {}

        #Retrieves existing brain databases
        #redis_storage = RedisStorage(self.redis)
        redis_storage = CRedisStorage(host=host, port=port)
        names = self.redis.lrange(BrainDatabaseManager.DATABASES_LIST_KEY, 0, -1)
        for name in names:
            lhash = pickle.loads(self.redis.hget(name, "hashing_config"))

            engine = Engine(lshashes=[lhash], storage=redis_storage)
            brain_database = BrainDatabase(name, self.redis, engine)
            self.brain_databases[name] = brain_database

    def __getitem__(self, name):
        return self.brain_databases.get(name, None)

    def new_brain_database(self, name, lhash, metadata={}):
        if name in self.brain_databases:
            raise ValueError("Brain database already exists: " + name)

        # Save general information about the new brain database
        self.redis.hset(name, "name", name)
        self.redis.hset(name, "nb_patches", 0)
        self.redis.hset(name, "nb_buckets", 0)
        self.redis.hset(name, "label_count_0", 0)
        self.redis.hset(name, "label_count_1", 0)

        # Save information about hashing function
        lhash.hash_name = name + "_" + lhash.hash_name
        self.redis.hset(name, "hashing_config", pickle.dumps(lhash))
        self.redis.hset(name, "hashing_name", lhash.hash_name)

        # Add new DB to the list of all DBs
        self.redis.rpush(BrainDatabaseManager.DATABASES_LIST_KEY, name)

        # Save information about metadata
        metadata_key = name + "_metadata"
        for attribute_name, attribute_info in metadata.items():
            self.redis.hset(metadata_key, attribute_name + "_dtype", attribute_info['dtype'])
            self.redis.hset(metadata_key, attribute_name + "_shape", attribute_info['shape'])

        redis_storage = RedisStorage(self.redis)
        engine = Engine(lshashes=[lhash], storage=redis_storage)
        brain_database = BrainDatabase(name, self.redis, engine)
        self.brain_databases[name] = brain_database

        return brain_database

    def remove_brain_database(self, brain_database):
        brain_database.engine.clean_all_buckets()
        self.redis.delete(brain_database.name)
        self.redis.delete(brain_database.name + "_metadata")
        self.redis.lrem(BrainDatabaseManager.DATABASES_LIST_KEY, brain_database.name)

    def remove_all_brain_databases(self):
        for brain_db in self.brain_databases.values():
            self.remove_brain_database(brain_db)

    def __contains__(self, name):
        return name in self.brain_databases

    def __str__(self):
        return "[" + ", ".join(sorted(self.brain_databases.keys())) + "]"

import pickle

import numpy as np
from nearpy import Engine
from nearpy.storage import storage_factory

from nearpy.filters import NearestFilter
from nearpy.data import NumpyData

from collections import defaultdict


class BrainDatabase(object):
    def __init__(self, name, storage, engine):
        self.name = name
        self.storage = storage
        self.engine = engine

        #Initialize metadata
        metadata = defaultdict(lambda: {})
        metadata_key = self.name + "_metadata"
        for key, value in self.storage.get_info(metadata_key).items():
            attribute_name, attribute_info = key.split('_')
            #if attribute_info == "shape":
            #    metadata[attribute_name][attribute_info] = eval(value)
            if attribute_info == "dtype":
                metadata[attribute_name][attribute_info] = np.dtype(value)
            else:
                metadata[attribute_name][attribute_info] = value

        self._metadata = {}
        for key, value in metadata.items():
            self._metadata[key] = NumpyData(key, value['dtype'], tuple(value['shape']))

    @property
    def metadata(self):
        return self._metadata

    def nb_patches(self, check_integrity=False):
        nb_patches = self.storage.get_info(self.name)["nb_patches"]
        nb_patches = int(nb_patches) if nb_patches is not None else 0

        if check_integrity:
            true_nb_patches = self.engine.nb_patches()
            if true_nb_patches != nb_patches:
                self.update(nb_patches=true_nb_patches, overwrite=True)
                return true_nb_patches

        return nb_patches

    def nb_buckets(self):
        return self.engine.nb_buckets()

    def buckets_size(self):
        return self.engine.buckets_size()

    def show_large_buckets(self, sizes, bucketkeys, use_spatial_code=False):
        from brainsearch import vizu
        indices = np.argsort(sizes)[::-1]
        #indices = range(len(sizes))

        #all_distances = []
        means = []
        stds = []

        nb_samples = 1000
        rng = np.random.RandomState(42)

        for idx in indices:
            print "{:,} neighbors".format(sizes[idx])
            patches = self.engine.storage.retrieve([bucketkeys[idx]], attribute=self.metadata['patch'])[0]
            labels = self.engine.storage.retrieve([bucketkeys[idx]], attribute=self.metadata['label'])[0]
            energies = np.sqrt(np.sum(patches**2, axis=tuple(range(1, patches.ndim))))

            indices = rng.randint(0, len(patches), 2*nb_samples)
            distances = np.sqrt(np.sum((patches[indices[1::2]] - patches[indices[::2]])**2, axis=tuple(range(1, patches.ndim))))

            #all_distances.append(distances)
            means.append(np.mean(distances))
            stds.append(np.std(distances))

            print means[-1]
            print stds[-1]

            #import pylab as plt
            #plt.hist(distances, bins=100)
            #plt.show()

            #print "0:{:,}, 1:{:,}".format(*np.bincount(labels.flatten()))
            #import pylab as plt
            #plt.hist(energies, bins=100)
            #plt.show()
            from ipdb import set_trace as dbg
            dbg()
            #vizu.show_images3d(patches, shape=self.metadata['patch'].shape, blocking=True)

        import pylab as plt
        plt.plot(means)
        plt.plot(stds)
        #plt.figure()
        #plt.hist(all_distances, bins=100)
        plt.show()

    def labels_count(self, check_integrity=False):
        info = self.storage.get_info(self.name)
        labels_count = np.array([info["label_count_0"], info["label_count_1"]])

        if check_integrity:
            true_labels_count = np.zeros(len(labels_count), dtype=self.metadata['label'].dtype)

            bucketkeys = self.engine.storage.bucketkeys()
            if len(bucketkeys) > 0:
                labels = self.engine.storage.retrieve(bucketkeys, attribute=self.metadata['label'])
                true_labels_count = np.bincount(np.concatenate(labels).flatten())

            if not np.all(true_labels_count == labels_count):
                self.update(labels_count=true_labels_count, overwrite=True)
                return true_labels_count

        return labels_count

    def insert(self, vectors, patches, labels, positions, brain_ids):
        data = {}
        data[self.metadata['patch']] = patches
        data[self.metadata['position']] = positions
        data[self.metadata['label']] = labels
        data[self.metadata['id']] = brain_ids

        hashkeys = self.engine.store_batch(vectors, data)
        self.update(nb_patches=len(vectors))
        self.update(labels_count=np.bincount(labels))
        return hashkeys

    def insert_with_pos(self, patches, labels, positions, brain_ids):
        data = {}
        data[self.metadata['label']] = labels
        data[self.metadata['id']] = brain_ids

        hashkeys = self.engine.store_batch_with_pos(patches, positions, data)
        self.update(nb_patches=len(patches))
        self.update(labels_count=np.bincount(labels))
        return hashkeys

    def get_neighbors(self, vectors, patches, attributes=None):
        if attributes is None:
            attributes = ['patch', 'label', 'position', 'id']

        for i, attribute in enumerate(attributes):
            attributes[i] = self.metadata[attribute]

        return self.engine.neighbors_batch(vectors, patches, *attributes)

    def get_neighbors_with_pos(self, patches, positions, radius, attributes=None):
        if attributes is None:
            attributes = ['patch', 'label', 'id']

        for i, attribute in enumerate(attributes):
            attributes[i] = self.metadata[attribute]

        return self.engine.neighbors_batch_with_pos(patches, positions, radius, *attributes)

    def update(self, nb_patches=None, labels_count=None, overwrite=False):
        info = self.storage.get_info(self.name)
        if nb_patches is not None:
            if overwrite:
                info["nb_patches"] = nb_patches
            else:
                info["nb_patches"] += nb_patches

        if labels_count is not None:
            for i, count in enumerate(labels_count):
                if overwrite:
                    info["label_count_{0}".format(i)] = int(count)
                else:
                    info["label_count_{0}".format(i)] += int(count)

        self.storage.set_info(self.name, info)

    def candidate_count(self, patches):
        candidate_count = self.engine.candidate_count_batch(patches)
        return candidate_count

    def query(self, data, k=None):
        if k is not None:
            self.engine.vector_filters = [NearestFilter(k)]

        return self.engine.neighbors_from_iter(data)


class BrainDatabaseManager(object):
    DATABASES_LIST_KEY = "BRAIN_DB"

    def __init__(self, storage_type, **storage_params):
        self.storage_type = storage_type
        self.storage_params = storage_params
        self.storage = storage_factory("file", **storage_params)
        self.brain_databases = {}

        #Retrieves existing brain databases
        names = self.storage.get_info(BrainDatabaseManager.DATABASES_LIST_KEY)
        for name in names:
            try:
                lhash = pickle.loads(self.storage.get_info(name)["hashing_config"])
                db_storage = storage_factory(storage_type, keyprefix=name, **storage_params)

                engine = Engine(lshashes=[lhash], storage=db_storage)
                brain_database = BrainDatabase(name, self.storage, engine)
                self.brain_databases[name] = brain_database
            except Exception as e:
                print "Cannot opened '{}'".format(name)
                print e.message[-100:]

    def __getitem__(self, name):
        return self.brain_databases.get(name, None)

    def new_brain_database(self, name, lhash, metadata={}):
        if name in self.brain_databases:
            raise ValueError("Brain database already exists: " + name)

        lhash.name = name + "_" + lhash.name

        # Save general information about the new brain database
        # and save information about hashing function
        self.storage.set_info(name, {"name": name,
                                     "nb_patches": 0,
                                     "nb_buckets": 0,
                                     "label_count_0": 0,
                                     "label_count_1": 0,
                                     "hashing_config": pickle.dumps(lhash),
                                     "hashing_name": lhash.name})

        # Add new DB to the list of all DBs
        self.storage.set_info(BrainDatabaseManager.DATABASES_LIST_KEY, name, append=True)

        # Save information about metadata
        metadata_key = name + "_metadata"
        metadata_dict = {}
        for attribute_name, attribute_info in metadata.items():
            metadata_dict[attribute_name + "_dtype"] = attribute_info['dtype']
            metadata_dict[attribute_name + "_shape"] = attribute_info['shape']

        self.storage.set_info(metadata_key, metadata_dict)

        db_storage = storage_factory(self.storage_type, keyprefix=name, **self.storage_params)
        engine = Engine(lshashes=[lhash], storage=db_storage)
        brain_database = BrainDatabase(name, self.storage, engine)
        self.brain_databases[name] = brain_database

        return brain_database

    def remove_brain_database(self, brain_database, full=False):
        if full:
            self.storage.del_info(brain_database.name)
            self.storage.del_info(brain_database.name + "_metadata")
            self.storage.del_info(BrainDatabaseManager.DATABASES_LIST_KEY, brain_database.name)
            brain_database.engine.storage.clear()
        else:
            brain_database.engine.clean_all_buckets()

    def remove_all_brain_databases(self, full=False):
        for brain_db in self.brain_databases.values():
            self.remove_brain_database(brain_db, full)

    def __contains__(self, name):
        return name in self.brain_databases

    def __str__(self):
        return "[" + ", ".join(sorted(self.brain_databases.keys())) + "]"

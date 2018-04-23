from collections import defaultdict
import os
import pickle

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

from model import prepare_model, set_random_seed, config_keras


REPO_PATH_DEL = "//"
PATH_COMMIT_DEL = "@"


def repo_dir(document):
    """
    Example of usage:
    >>> repo_path('github.com/some/repo.git//path/to/file.java@commit')
    'github.com/some/repo.git//path/to'

    :param doc: string in the format 'repo//path@commit'
    :return: `repo//path` without filename
    """
    repo = repo_from_doc(document)
    directory = dir_from_doc(document)
    return REPO_PATH_DEL.join([repo, directory])


def repo_from_doc(document):
    return document.split(REPO_PATH_DEL)[0]


def path_from_doc(document):
    return document.split(REPO_PATH_DEL)[1].split(PATH_COMMIT_DEL)[0]


def dir_from_doc(document):
    return path_from_doc(document).rsplit("/", 1)[0]


def docs_to_groups(documents):
    """
    Creates groups of 'similar' files - files in the same directory from the same repository.
    :param documents: list of strings in the format 'repo//path@commit'
    :return: dictionary that stores 'repo//path/to/dir' and related indices of documents
    """
    groups = defaultdict(list)
    for i, doc in enumerate(tqdm(documents)):
        groups[repo_dir(doc)].append(i)

    return groups


def docs_to_repo_groups(documents):
    """
    Creates groups of 'similar' files - files in the same directory from the same repository.
    :param documents: list of strings in the format 'repo//path@commit'
    :return: dictionary that stores 'repo' that stores dict with key 'path/to/dir' and related
             indices of documents
    """
    groups = defaultdict(lambda: defaultdict(list))
    for i, doc in enumerate(tqdm(documents)):
        groups[repo_from_doc(doc)][dir_from_doc(doc)].append(i)

    return groups


def sample_group(group, n_samples):
    res = []
    for i in range(n_samples):
        a, b = np.random.choice(len(group), size=2, replace=False)
        res.append((a, b))
    return res


def repo_sampling(groups, n_pos, n_neg, threshold=2, n_samples_per_group=1):
    good_groups = list(filter(lambda group: len(group) >= threshold, groups))
    groups_ = list(groups)

    if not good_groups:
        # no big groups
        return None

    pos_pairs = []
    while len(pos_pairs) < n_pos:
        group_id = np.random.choice(len(good_groups))
        pos_pairs.extend(sample_group(good_groups[group_id], n_samples_per_group))
    pos_pairs = pos_pairs[:n_pos]

    neg_pairs = []
    while len(neg_pairs) < n_neg:
        group_id_a, group_id_b = np.random.choice(len(groups), size=2)
        neg_pairs.append((np.random.choice(groups_[group_id_a]),
                          np.random.choice(groups_[group_id_b])))
    neg_pairs = neg_pairs[:n_neg]

    labels = [1] * n_pos + [0] * n_neg
    pairs = pos_pairs + neg_pairs

    return pairs, labels


def make_generator(repo_groups, matrix, batch_size=512, samples_per_repo=4):
    repos = list(repo_groups.keys())

    def generator():
        while True:
            pairs = []
            labels = []

            while len(pairs) < batch_size:
                # positive and negative samples from the same repo
                n_pos, n_neg = samples_per_repo, samples_per_repo

                repo = np.random.choice(repos)
                res = repo_sampling(repo_groups[repo].values(), n_pos, n_neg)
                if res is not None:
                    pairs.extend(res[0])
                    labels.extend(res[1])

                # positive samples from the same repo
                n_pos, n_neg = max(1, samples_per_repo // 2), 0

                repo = np.random.choice(repos)
                res = repo_sampling(repo_groups[repo].values(), n_pos, n_neg)
                if res is not None:
                    pairs.extend(res[0])
                    labels.extend(res[1])

                # negative samples from the same repo
                n_pos, n_neg = 0, max(1, samples_per_repo // 2)

                repo = np.random.choice(repos)
                res = repo_sampling(repo_groups[repo].values(), n_pos, n_neg)
                if res is not None:
                    pairs.extend(res[0])
                    labels.extend(res[1])

            pairs = np.array(pairs)
            labels = np.array(labels)

            # shuffle
            ind = np.arange(pairs.shape[0])
            np.random.shuffle(ind)

            # make batch
            pairs = pairs[ind][:batch_size]
            labels = labels[ind][:batch_size]

            yield [matrix[pairs[:, 0]], matrix[pairs[:, 1]]], labels
    return generator()


def pipeline(data_loc, test_size=0.1, batch_size=512, val_batch_size=2000,
             samples_per_repo=4, seed=1989,
             hidden_sizes=[32, 64, 128], activation="relu", dropout_rate=0.1,
             optimizer="rmsprop", steps_per_epoch=1000, validation_steps=100, max_queue_size=10,
             workers=4, epochs=10):
    set_random_seed(seed)
    config_keras()

    # load data from pickle
    with open(os.path.join(data_loc), "rb") as f:
        data = pickle.load(f)
        matrix = data["matrix"]
        documents = data["documents"]
    print("Number of documents:", len(documents))
    print("Matrix shape:", matrix.shape)

    # prepare groups
    doc_groups = docs_to_groups(documents)
    # some statistics
    print("Number of groups:", len(doc_groups))
    print("Number of groups with 1 element:",
          len(list(filter(lambda r: len(r) == 1, doc_groups.values()))))
    for i in range(1, 6):
        print("Number of groups bigger than {}: {}".format(i,
                                                           len(list(filter(lambda r: len(r) > i,
                                                                           doc_groups.values())))))

    repo_groups = docs_to_repo_groups(documents)
    # some statistics
    print("Number of repos:", len(repo_groups))

    # train / test split
    train_repos, test_repos = train_test_split(list(repo_groups.keys()), test_size=test_size)
    print("Number of train repositories: {}, number of test repositories: {}"
          .format(len(train_repos), len(test_repos)))

    train_groups = {}
    for repo in tqdm(train_repos):
        train_groups[repo] = repo_groups[repo]

    test_groups = {}
    for repo in tqdm(test_repos):
        test_groups[repo] = repo_groups[repo]

    # prepare generators
    train_gen = make_generator(train_groups, matrix, batch_size=batch_size,
                               samples_per_repo=samples_per_repo)
    test_gen = make_generator(test_groups, matrix, batch_size=val_batch_size,
                              samples_per_repo=samples_per_repo)

    model, emb_model_a = prepare_model(input_shape=matrix.shape[1], hidden_sizes=hidden_sizes,
                                       activation=activation, dropout_rate=dropout_rate,
                                       optimizer=optimizer)
    model.summary()

    model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, validation_data=test_gen,
                        validation_steps=validation_steps, max_queue_size=max_queue_size,
                        workers=workers, epochs=epochs)

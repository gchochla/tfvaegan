#import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import os
import sys
import pdb
# import h5py

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1

        with open(os.path.join(opt.dataroot, opt.dataset, 'trainclasses1.txt')) as fp:
            train_class_names = [line.strip() for line in fp.readlines()]
        with open(os.path.join(opt.dataroot, opt.dataset, 'valclasses1.txt')) as fp:
            val_class_names = [line.strip() for line in fp.readlines()]

        train_loc, val_unseen_loc = [], []
        for i, class_name in enumerate(matcontent['allclasses_names']):
            class_name = class_name[0][0]
            if class_name in train_class_names:
                train_loc.append(i)
            elif class_name in val_class_names:
                val_unseen_loc.append(i)

        train_loc = np.array(train_loc)
        val_unseen_loc = np.array(val_unseen_loc)
        # train_loc = matcontent['train_loc'].squeeze() - 1
        # val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),self.attribute.size(1))

        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long() 

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))


        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 

    def next_seen_batch(self, seen_batch):
        idx = torch.randperm(self.ntrain)[0:seen_batch]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_att

class MatDataset:
    """Multipurpose dataset based on the .mat files
    provided by Xian et al., 2017.

    Attributes:
        benchmark(str): benchmark.
        attributes(torch.Tensor): row-wise attributes. Indexed with
            actual label of class (-1 because real labels start at 1)
        train_features(torch.Tensor): row-wise ResNet101 features to
            be used for training.
        eval_seen_features(torch.Tensor): row-wise ResNet101 features
            to be used for evaluation of seen classes accuracy. Only
            included if `generalized` is set to `True`.
        eval_unseen_features(torch.Tensor): row-wise ResNet101 features
            to be used for evaluation of unseen classes accuracy. Only
            included if `generalized` is set to `True` at init.
        eval_features(torch.Tensor): row-wise ResNet101 features to be
            used fot evaluation of accyracy. Only included if
            `generalized` is set to `False` at init.
        train_labels(numpy.ndarray): actual labels corresponding to
            `train_features`.
        eval_seen_labels(numpy.ndarray): actual labels corresponding to
            `eval_seen_features`.
        eval_unseen_labels(numpy.ndarray): actual labels corresponding to
            `eval_unseen_features`.
        eval_labels(numpy.ndarray): actual labels corresponding to
            `eval_features`.
        train_label_mapping(dict): maps actual training labels to usable
            labels.
        eval_label_mapping(dict): maps actual evaluation labels to usable
            labels.
        generalized(bool): whether this dataset is to be used in the GZSL
            setting.
        training(bool): what state the dataset is currently in.
        class_mapping(dict): contains indices to corresponding feature
            tensors of each class.
    """

    def __init__(self, dataset_dir, benchmark, l2_norm=True,
                 validation=True, norm=True, generalized=False):
        """Init.

        Args:
            dataset_dir(str): path to directory provided
                by Xian et al., 2017.
            benchmark(str): which benchmark dataset to use
                (as they appear in `dataset_dir`, e.g. 'CUB').
            l2_norm(bool, optional): whether to apply L2
                normalization on the attributes, default `True`
                (recommended).
            validation(bool, optional): basically whether dataset
                is used for hyperparameter search, default `True`.
            norm(bool, optional): whether to project project the
                features to [0, 1] (based on the current training
                features), default `True`.
            generalized(bool, optional): whether to split the features
                for Generalized ZSL or not, default `False`.
        """

        self.benchmark = benchmark

        matcontent = sio.loadmat(os.path.join(dataset_dir, 'data',
                                              benchmark.upper(),
                                              'res101.mat'))
        features = matcontent['features'].T.astype(np.float32)

        ##################
        ### get labels ###

        indices = matcontent['labels'].squeeze() - 1

        # get corresponding labels
        cls_fn = os.path.join(dataset_dir, 'data',
                              benchmark.upper(),
                              'allclasses.txt')
        with open(cls_fn, 'r') as avcls:
            labels = {i: line.strip() for i, line in enumerate(avcls.readlines())}

        ### get labels ###
        ##################

        matcontent = sio.loadmat(os.path.join(dataset_dir, 'data',
                                              benchmark.upper(),
                                              'att_splits.mat'))

        ######################
        ### get attributes ###

        if l2_norm:
            attributes = matcontent['att'].T
        else:
            attributes = matcontent['original_att'].T

        self.attributes = torch.FloatTensor(attributes)

        ### get attributes ###
        ######################

        if validation:
            # trash test features and labels
            trainval_loc = matcontent['trainval_loc'].squeeze() - 1
            test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
            trainval_loc = np.concatenate((trainval_loc, test_seen_loc))

            features = features[trainval_loc]
            indices = indices[trainval_loc]

            # get train labels to separate train and validation
            cls_fn = os.path.join(dataset_dir, 'data',
                                  benchmark.upper(),
                                  'trainclasses1.txt')
            with open(cls_fn, 'r') as avcls:
                trainclasses = [line.strip() for line in avcls.readlines()]

            train_loc, val_loc = [], []
            for i in range(len(features)):
                if labels[indices[i]] in trainclasses:
                    train_loc.append(i)
                else:
                    val_loc.append(i)

            # arbitrary split for validation
            split = int(0.8 * len(train_loc))  # does not guarantee samples from all classes
            train_loc, test_seen_loc = train_loc[:split], train_loc[split:]

            train_features = features[train_loc]
            eval_seen_features = features[test_seen_loc]
            eval_unseen_features = features[val_loc]

            train_labels = indices[train_loc]
            eval_seen_labels = indices[test_seen_loc]
            eval_unseen_labels = indices[val_loc]

        else:
            trainval_loc = matcontent['trainval_loc'].squeeze() - 1
            test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
            test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

            train_features = features[trainval_loc]
            eval_seen_features = features[test_seen_loc]
            eval_unseen_features = features[test_unseen_loc]

            train_labels = indices[trainval_loc]
            eval_seen_labels = indices[test_seen_loc]
            eval_unseen_labels = indices[test_unseen_loc]

        if generalized:
            if norm:
                mmscaler = preprocessing.MinMaxScaler()
                train_features = mmscaler.fit_transform(train_features)
                eval_seen_features = mmscaler.transform(eval_seen_features)
                eval_unseen_features = mmscaler.transform(eval_unseen_features)

            self.train_features = torch.from_numpy(train_features)
            self.eval_seen_features = torch.from_numpy(eval_seen_features)
            self.eval_unseen_features = torch.from_numpy(eval_unseen_features)

            self.train_labels = train_labels
            self.eval_seen_labels = eval_seen_labels
            self.eval_unseen_labels = eval_unseen_labels

            # map actual labels to usable (sequential) labels
            self.train_label_mapping = {int(label): i for i, label in
                                        enumerate(set(self.train_labels))}
            self.eval_label_mapping = {int(label): i for i, label in
                                       enumerate(set(self.eval_seen_labels))}
            self.eval_label_mapping.update({
                int(label): i + len(self.eval_label_mapping) for i, label in
                enumerate(set(self.eval_unseen_labels))
            })
        else:
            train_features = np.concatenate((train_features, eval_seen_features))
            if norm:
                mmscaler = preprocessing.MinMaxScaler()
                train_features = mmscaler.fit_transform(train_features)
                eval_unseen_features = mmscaler.transform(eval_unseen_features)

            self.train_features = torch.from_numpy(train_features)
            self.eval_features = torch.from_numpy(eval_unseen_features)

            train_labels = np.concatenate((train_labels, eval_seen_labels))
            self.train_labels = train_labels
            self.eval_labels = eval_unseen_labels

            # map actual labels to usable (sequential) labels
            self.train_label_mapping = {int(label): i for i, label in
                                        enumerate(set(self.train_labels))}
            self.eval_label_mapping = {int(label): i for i, label in
                                       enumerate(set(self.eval_labels))}

        self.generalized = generalized
        self.training = True
        self._init_class_mapping()

    @property
    def n_classes(self):
        """Returns:
            Number of current classes.
        """
        return len(self.class_mapping[self.training])

    def train(self):
        """Sets dataset to training mode.

        Returns:
            Self.
        """
        self.training = True
        return self

    def eval(self):
        """Sets dataset to evaluation mode.

        Returns:
            Self.
        """
        self.training = False
        return self

    def __len__(self):
        """Returns length of current features."""
        if self.training:
            return len(self.train_features)
        if self.generalized:
            return len(self.eval_seen_features) + len(self.eval_unseen_features)
        return len(self.eval_features)

    def __getitem__(self, index):
        """Fetches info according to setting.

        Args:
            index(int): index.

        Returns:
            If ZSL or in training mode: feature, usable label and
            attribute. Else, if GZSL and in evaluation mode: feature,
            usable label, attribute and whether the class is an unseen one.
        """

        if not -len(self) <= index < len(self):
            raise IndexError('index {} is out of bounds for size {}'.format(index, len(self)))

        if index < 0:
            index += len(self)

        if self.training:
            lbl_ind = self.train_labels[index]
            return (self.train_features[index],
                    self.train_label_mapping[lbl_ind],
                    self.attributes[lbl_ind])

        if self.generalized:
            if index < len(self.eval_seen_features):
                lbl_ind = self.eval_seen_labels[index]
                return (self.eval_seen_features[index],
                        self.eval_label_mapping[lbl_ind],
                        self.attributes[lbl_ind], False)

            index -= len(self.eval_seen_features)
            lbl_ind = self.eval_unseen_labels[index]
            return (self.eval_unseen_features[index],
                    self.eval_label_mapping[lbl_ind],
                    self.attributes[lbl_ind], True)

        lbl_ind = self.eval_labels[index]
        return (self.eval_features[index],
                self.eval_label_mapping[lbl_ind],
                self.attributes[lbl_ind])

    def __call__(self, way=None, queries_per=None):
        """Fetches a randomly sampled episode.

        Args:
            way(int|NoneType): number of classes,
                default `None`(all).
            queries_per(int|NoneType): number of samples
                per class, default `None`(all).

        Returns:
            If in training mode: a list of torch.Tensors with
            `queries_per` features per class, the attributes and
            lastly a list of corresponding usable labels. If in
            evaluation mode: a list of torch.Tensors with
            `queries_per` features per class, the attributes, a list of
            corresponding usable labels and the number of seen classes.
            Seen classes are at the start of the lists.
        """

        query = []
        labels = []

        if self.training:
            rand_labels_inds = torch.randperm(len(self.train_label_mapping))[:way]
            # int64 so it doesn't get interpreted as bool
            rand_labels = torch.LongTensor(list(self.train_label_mapping))[
                rand_labels_inds
            ]
            attributes = self.attributes[rand_labels]

            labels = [self.train_label_mapping[lbl] for lbl in rand_labels]

            for lbl in rand_labels:
                inds = self.class_mapping[True][lbl]
                rand_inds = torch.randperm(len(inds))[:queries_per]
                query.append(self.train_features[rand_inds])

            return query, attributes, labels

        # if not training

        rand_labels_inds = torch.randperm(len(self.eval_label_mapping))[:way]
        rand_labels = np.array(list(self.eval_label_mapping),
                               dtype=np.int64)[rand_labels_inds]
        # order labels (seen then unseen)
        labels = []
        unseen_labels = []
        for lbl in rand_labels:
            try:
                if lbl in self.eval_unseen_labels:
                    unseen_labels.append(lbl)
                else:
                    labels.append(lbl)
            except AttributeError:
                unseen_labels.append(lbl)

        seen_classes = len(labels)

        labels.extend(unseen_labels)
        attributes = self.attributes[torch.LongTensor(labels)]

        query = []

        for i, lbl in enumerate(labels):
            inds = self.class_mapping[False][lbl]
            rand_inds = torch.randperm(len(inds))[:queries_per]
            if self.generalized:
                if i < seen_classes:
                    query.append(self.eval_seen_features[rand_inds])
                else:
                    query.append(self.eval_unseen_features[rand_inds])
            else:
                query.append(self.eval_features[rand_inds])

        labels = [self.eval_label_mapping[lbl] for lbl in labels]

        return query, attributes, labels, seen_classes

    def _init_class_mapping(self):
        """Proporly sets `class_mapping`."""

        self.class_mapping = {True: dict(), False: dict()}  # why bool? whether training

        for i, lbl in enumerate(self.train_labels):
            self.class_mapping[True].setdefault(lbl, []).append(i)

        if self.generalized:
            for i, lbl in enumerate(self.eval_unseen_labels):
                self.class_mapping[False].setdefault(lbl, []).append(i)
            for i, lbl in enumerate(self.eval_seen_labels):
                self.class_mapping[False].setdefault(lbl, []).append(i)
        else:
            for i, lbl in enumerate(self.eval_labels):
                self.class_mapping[False].setdefault(lbl, []).append(i)

        for training in self.class_mapping:
            for lbl in self.class_mapping[training]:
                self.class_mapping[training][lbl] = \
                    np.array(self.class_mapping[training][lbl])

    def fsl_episode(self, way, shot, queries_per):
        """Fetches and FSL episode.

        Args:
            way(int): number of classes.
            shot(int): number of samples per class for
                support.
            queries_per(int): number of samples per class
                for querying.

        Returns:
            `list` of `way` `torch.Tensor`s with `shot` vectors each,
            `list` of `way` `torch.Tensor`s with `queries_per` vectors
            each and `list` of `way` `int`s with usable labels.
        """

        features, _, class_ids, *unseen_classes = self(
            way=way, queries_per=shot+queries_per
        )
        query = [feats[:queries_per] for feats in features]
        support = [feats[queries_per:] for feats in features]
        return (support, query, class_ids, *unseen_classes)


def load_model(model, path):
    """Loads model for old PyTorch version.

    Args:
        model (`nn.Module`): Net whose parameters are loaded.
        path (`str`): path to parameters.
    """
    data_dict = {}
    fin = open(path, 'r')
    i = 0
    odd = 1
    prev_key = None
    while True:
        s = fin.readline().strip()
        if not s:
            break
        if odd:
            prev_key = s
        else:
            print('Iter', i)
            val = eval(s)
            if type(val) != type([]):
                data_dict[prev_key] = torch.FloatTensor([eval(s)])[0]
            else:
                data_dict[prev_key] = torch.FloatTensor(eval(s))
            i += 1
        odd = (odd + 1) % 2

    # Replace existing values with loaded
    own_state = model.state_dict()
    print('Items:', len(own_state.items()))
    for k, v in data_dict.items():
        if not k in own_state:
            print('Parameter', k, 'not found in own_state!!!')
        else:
            try:
                own_state[k].copy_(v)
            except:
                print('Key:', k)
                print('Old:', own_state[k])
                print('New:', v)
                sys.exit(0)
    print('Model loaded')


def tensor_interleave(tensor, times):
    interleave = []
    for row in tensor:
        interleave.extend([row] * times)
    return torch.stack(interleave)

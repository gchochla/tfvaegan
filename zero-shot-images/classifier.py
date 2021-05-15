#author: akshitac8
from typing import no_type_check_decorator
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler 
import sys
import copy
import pdb

class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True, netDec=None, dec_size=4096, dec_hidden_size=4096):
        self.train_X =  _train_X.clone() 
        self.train_Y = _train_Y.clone() 
        self.test_seen_feature = data_loader.test_seen_feature.clone()
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature.clone()
        self.test_unseen_label = data_loader.test_unseen_label 
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
        self.netDec = netDec
        if self.netDec:
            self.netDec.eval()
            self.input_dim = self.input_dim + dec_size
            self.input_dim += dec_hidden_size
            self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
            self.train_X = self.compute_dec_out(self.train_X, self.input_dim)
            self.test_unseen_feature = self.compute_dec_out(self.test_unseen_feature, self.input_dim)
            self.test_seen_feature = self.compute_dec_out(self.test_seen_feature, self.input_dim)
        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size) 
        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        if generalized:
            self.acc_seen, self.acc_unseen, self.H, self.epoch= self.fit()
            #print('Final: acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (self.acc_seen, self.acc_unseen, self.H))
        else:
            self.acc,self.best_model = self.fit_zsl() 
            #print('acc=%.4f' % (self.acc))
    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        last_loss_epoch = 1e8 
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                mean_loss += loss.data[0]
                loss.backward()
                self.optimizer.step()
                #print('Training classifier loss= ', loss.data[0])
            acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            #print('acc %.4f' % (acc))
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(self.model.state_dict())
        return best_acc, best_model 
        
    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        out = []
        best_model = copy.deepcopy(self.model.state_dict())
        # early_stopping = EarlyStopping(patience=20, verbose=True)
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
            acc_seen = 0
            acc_unseen = 0
            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
        return best_seen, best_unseen, best_H,epoch
                     
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]


    def val_gzsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda(), volatile=True)
            else:
                inputX = Variable(test_X[start:end], volatile=True)
            output = self.model(inputX)  
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        acc_per_class /= target_classes.size(0)
        return acc_per_class 

    # test_label is integer 
    def val(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda(), volatile=True)
            else:
                inputX = Variable(test_X[start:end], volatile=True)
            output = self.model(inputX) 
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        return acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        return acc_per_class.mean() 


    def compute_dec_out(self, test_X, new_size):
        start = 0
        ntest = test_X.size()[0]
        new_test_X = torch.zeros(ntest,new_size)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda(), volatile=True)
            else:
                inputX = Variable(test_X[start:end], volatile=True)
            feat1 = self.netDec(inputX)
            feat2 = self.netDec.getLayersOutDet()
            new_test_X[start:end] = torch.cat([inputX,feat1,feat2],dim=1).data.cpu()
            start = end
        return new_test_X


class LINEAR_LOGSOFTMAX_CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX_CLASSIFIER, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o


def init_fn(mod):
    """Initializes linear layers "diagonally"
    (concerning [:in_features, :in_features]).
    Function to pass to .apply()."""
    classname = mod.__class__.__name__
    if classname.find('Linear') != -1:
        init = torch.randn(mod.weight.size()) / 10
        init[range(mod.in_features), range(mod.in_features)] = 1
        mod.weight = nn.Parameter(init, requires_grad=True)
        if mod.bias is not None:
            mod.bias = nn.Parameter(
                torch.zeros_like(mod.bias).data,
                requires_grad=True
            )

class MLP(nn.Module):
    """Simple MLP.

    Attributes:
        layers(nn.Module): sequence of layers.
    """

    def __init__(self, in_features, out_features, hidden_layers=None, dropout=0,
                 hidden_actf=nn.LeakyReLU(0.2), output_actf=nn.ReLU()):
        """Init.

        Args:
            in_features(int): input dimension.
            out_features(int): final output dimension.
            hidden_layers(list of ints|int|None, optional): list o
                hidden layer sizes of arbitrary length or int for one
                hidden layer, default `None` (no hidden layers).
            dropout(float, optional): dropout probability, default
                `0`.
            hidden_actf(activation function, optional): activation
                function of hidden layers, default `nn.LeakyReLU(0.2)`.
            output_actf(activation function, optional): activation
                function of output layers, default `nn.ReLU()`.
            noise_std(float, optional): std dev of gaussian noise to add
                to MLP result, default `0` (no noise).
        """

        if hidden_layers is None:
            hidden_layers = []
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]

        super().__init__()

        hidden_layers = [in_features] + hidden_layers + [out_features]

        layers = []
        for i, (in_f, out_f) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
            layers.append(nn.Linear(in_f, out_f))

            if i != len(hidden_layers) - 2:
                # up to second-to-last layer
                layers.append(hidden_actf)
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(output_actf)  # ok to use relu, resnet feats are >= 0

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward propagation.

        Args:
            x(torch.Tensor): input of size (batch, in_features).

        Returns:
            A torch.Tensor of size (batch, out_features).
        """
        return self.layers(x)

    def init_diagonal(self):
        """Sets weights of linear layers to approx I
        and biases to 0.
        """
        self.apply(init_fn)


class PrototypicalNet(nn.Module):
    """Classifies examples based on distance metric
    from class prototypes. FSL setting.

    Attributes:
        mapper(nn.Module): mapper from feature to
            embedding space.
        dist(function): distance function. Accepts
            2D torch.Tensor prototypes and 2D
            torch.Tensor queries and returns a 2D torch.Tensor
            whose [i, j] element is the distance between
            queries[i] nad prototypes[j].
    """

    def __init__(self, in_features, out_features, hidden_layers=None,
                 dist='euclidean', init_diagonal=False, **extra_features):
        """Init.

        Args:
            in_features(int): input features dimension.
            out_features(int): final output dimension.
            extra_features(int): extra feature dimension,
                included for the model to be backwards
                compatible with pretrained models without
                extra features. DEfault is None, aka no extra
                features.
            hidden_layers(list|None): number of neurons
                in hidden layers, default is one hidden
                layer with units same as input.
            dist(function|str): distance metric. If str,
                predefined distance is used accordingly.
                If function is passed, it should accept
                2D torch.Tensor prototypes and 2D
                torch.Tensor queries and return a 2D torch.Tensor
                whose [i, j] element is the distance between
                queries[i] nad prototypes[j].
            init_diagonal(bool): whether to init linear layers
                with diagonal weights and zero biases, default=`False`.
        """

        super().__init__()

        if hidden_layers is None:
            hidden_layers = [in_features]

        self.mapper = MLP(in_features, out_features, hidden_layers,
                          hidden_actf=nn.ReLU())

        if extra_features:
            extra_dim = extra_features['dim']
            extra_layers = [extra_dim] + extra_features.get(
                'hidden_layers', [extra_dim] * len(hidden_layers)
            ) + [extra_features.get('out_dim', extra_dim)] 

            self.mapper_from_extra = nn.ModuleList(
                [
                    nn.Linear(in_feats, out_feats + extra_out_feats)
                    for in_feats, (out_feats, extra_out_feats) in zip(
                        extra_layers[:-1], zip(
                            hidden_layers + [out_features], extra_layers[1:]
                        )
                    )
                ]
            )
            self.mapper_to_extra = nn.ModuleList(
                [
                    nn.Linear(in_feats, out_feats)
                    for in_feats, out_feats in zip(
                        [in_features] + hidden_layers, extra_layers[1:]
                    )
                ]
            )
        else:
            self.mapper_from_extra = nn.Identity()
            self.mapper_to_extra = nn.Identity()

        if init_diagonal:
            self.mapper.init_diagonal()
            if extra_features:
                self.mapper_from_extra.apply(init_fn)

        if isinstance(dist, str):
            self.dist = self.__getattribute__(dist)
        else:
            self.dist = dist

    @staticmethod
    def cosine(prototypes, queries):
        """Computes cosine distance between prototypes
        and set of queries.

        Args:
            prototypes(torch.Tensor): prototypes of size
                (way, embedding_dim).
            queries(torch.Tensor): queries of size
                (n_queries, embedding_dim).

        Returns:
            A torch.Tensor of size (n_queries, way) where
            element [i,j] contains distance between queries[i]
            and prototypes[j].
        """

        inner_prod = queries.matmul(prototypes.T)
        norm_i = queries.norm(dim=1, keepdim=True)
        norm_j = prototypes.norm(dim=1, keepdim=True).T
        return 1 - inner_prod / norm_i / norm_j

    @staticmethod
    def euclidean(prototypes, queries):
        """Computes euclidean distance between prototypes
        and set of queries.

        Args:
            prototypes(torch.Tensor): prototypes of size
                (way, embedding_dim).
            queries(torch.Tensor): queries of size
                (n_queries, embedding_dim).

        Returns:
            A torch.Tensor of size (n_queries, way) where
            element [i,j] contains distance between queries[i]
            and prototypes[j].
        """
        way = prototypes.size(0)
        n_queries = queries.size(0)

        prototypes = prototypes.repeat(n_queries, 1)
        queries = util.tensor_interleave(queries, way)
        # after the repeats, prototypes have way classes after way classes after ...
        # and queries have way repeats of 1st query, way repeats of 2nd query, ...
        # so initial dist vector has distance of first query to all way classes
        # then the distance of the second query to all way class, etc
        return torch.norm(prototypes - queries, dim=1).view(n_queries, way)

    def map(self, main_tensor, extra_tensor1, extra_tensor2):

        extra_tensor = torch.cat((extra_tensor1, extra_tensor2), dim=1)

        for layer, (m_from, m_to) in enumerate(
            zip(self.mapper_from_extra, self.mapper_to_extra)
        ):
            mapper = self.mapper[3 * layer: 3 * (layer + 1)]
            from_extra = m_from(extra_tensor)
            to_extra = m_to(main_tensor)
            main_z = mapper[0](main_tensor)

            extra_tensor = mapper[1:](
                to_extra + from_extra[:, :to_extra.size(1), :to_extra.size(2)]
            )
            main_tensor = mapper[1:](
                main_z + from_extra[:, to_extra.size(1):, to_extra.size(2):]
            )

        return torch.cat((main_tensor, extra_tensor), dim=1)


    def forward(self, support, query, netDec):
        """Episodic forward propagation.

        Computes prototypes given the support set of an episode
        and then makes inference on the corresponding query set.

        Args:
            support(list of torch.Tensors): support set list
                whose every element is tensor of size
                (shot, feature_dim), i.e. shot image features
                belonging to the same class.
            query(list of torch.Tensors): query set list
                whose every element is tensor of size
                (n_queries, feature_dim), i.e. n_queries image
                features belonging to the same class (for consistency
                purposes with support).
            mix(bool): mix support and query, default=`False`.

        Returns:
            A list of torch.Tensor of size (n_queries, way) logits
            whose i-th element consists of logits of queries belonging
            to the i-th class.
        """

        prototypes = []
        for class_features in support:
            # class_features are (shot, feature_dim)
            feat1 = netDec(class_features)
            feat2 = netDec.getLayersOutDec()
            prototypes.append(self.map(class_features, feat1, feat2).mean(dim=0))
        prototypes = torch.stack(prototypes)

        logits = []
        for class_features in query:
            # class_features are (n_queries, feature_dim)
            feat1 = netDec(class_features)
            feat2 = netDec.getLayersOutDec()
            logits.append(
                -self.dist(prototypes, self.map(class_features, feat1, feat2))
            )

        return logits

def eval_protonet(fsl_classifier, netDec, dataset, support, labels, cuda):
    """Return ZSL or GZSL metrics of Z2FSL.

    Args:
        fsl_classifier (`nn.Module`): trained `PrototypicalNet`-like
            classifier.
        dataset (`MatDataset`): dataset used during training.
        support (`list` of `torch.Tensor`s): support set.
        labels (`iterable` of `int`s): corresponding labels, one
            for each element in `support`.
        cuda (`bool`): whether on CUDA.

    Returns:
        If ZSL:
            `float`: ZSL accuracy.
        Else:
            `float`: harmonic mean.
            `float`: seen accuracy.
            `float`: unseen accuracy.
    """

    fsl_classifier.eval()
    dataset.eval()

    query, _, align_labels, n_seen = dataset()
    query = [Variable(query[align_labels.index(label)]) for label in labels]
    if cuda:
        query = [cls_query.cuda() for cls_query in query]

    logits = fsl_classifier(support, query, netDec)

    fsl_classifier.train()
    dataset.train()

    accs = []
    for i, class_logits in enumerate(logits):
        preds = np.argmax(class_logits.data.cpu(), -1)
        correct = (preds == i).sum()
        accs.append(correct / preds.size(0))

    if n_seen > 0:
        acc_s = sum(accs[:n_seen]) / n_seen
        acc_u = sum(accs[n_seen:]) / (len(accs) - n_seen)
        acc = 2 * acc_s * acc_u / (acc_s + acc_u)
        return [acc, acc_s, acc_u]
    else:
        return sum(accs) / len(accs)

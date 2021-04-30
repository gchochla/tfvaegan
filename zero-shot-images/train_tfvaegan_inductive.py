#author: akshitac8
#tf-vaegan inductive
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import math
import sys
from sklearn import preprocessing
import csv
#import functions
import model
import util
import classifier as classifier
from config import opt

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# load data
## BEFORE
# data = util.DATA_LOADER(opt)
# print("# of training samples: ", data.ntrain)
## AFTER
data = util.MatDataset(dataset_dir=os.path.join(opt.dataroot, os.path.pardir),
                       benchmark=opt.dataset, validation=opt.validation,
                       norm=opt.preprocessing, generalized=opt.gzsl)
shot = opt.shot
queries = opt.queries
print('# of training samples: ', len(data))
##

netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator_D1(opt)
# Init models: Feedback module, auxillary module
netF = model.Feedback(opt)
netDec = model.AttDec(opt,opt.attSize)
clsf = classifier.PrototypicalNet(in_features=opt.resSize, out_features=opt.resSize,
                                  init_diagonal=True, hidden_layers=[opt.resSize] * opt.fsl_num_layers)
if opt.fsl_directory is not None:
    model_path = os.path.join(
        opt.fsl_directory,
        f'fsl_{data.benchmark.lower()}_features_proposed_splits_'
        f'train{"" if opt.validation else "val"}'
        f'{"_generalized" if opt.gzsl else ""}.pt'
    )
    print(f'Loading model from {model_path}')
    util.load_model(clsf, model_path)

print(netE)
print(netG)
print(netD)
print(netF)
print(netDec)
print(clsf)

###########
# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize) #attSize class-embedding size
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1
##########
# Cuda
if opt.cuda:
    netD.cuda()
    netE.cuda()
    netF.cuda()
    netG.cuda()
    netDec.cuda()
    clsf.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()

class EpisodeCrossEntropyLoss(nn.Module):
    """FSL episode classification loss.

    Attributes:
        criterion(nn.Module): module that computes loss.
        reduction(str): how to reduce vector results.
    """
    def __init__(self, reduction='mean'):
        """Init.

        Args:
            reduction(str, optional): how to reduce
                vector results, default `'mean'`.
        """

        assert reduction in ('mean', 'sum')

        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.reduction = reduction

    def forward(self, logits):
        """Computes loss.

        Args:
            logits(list): list of torch.Tensor logits. Index
                i corresponds to i-th class.

        Returns:
            Loss.
        """

        loss = []
        for i, class_logits in enumerate(logits):
            labels = i * torch.LongTensor([1] * class_logits.size(0))
            if opt.cuda:
                labels = labels.cuda()
            labelsv = Variable(labels)
            loss.append(self.criterion(class_logits, labelsv))

        if self.reduction == 'mean':
            return sum(loss) / len(loss)
        elif self.reduction == 'sum':
            return sum(loss)

clsf_loss = EpisodeCrossEntropyLoss()

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),size_average=False)
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (BCE + KLD)

def sample():
    ## BEFORE
    # batch_feature, batch_att = data.next_seen_batch(opt.batch_size)
    ## AFTER
    batch_feature, batch_att, *_ = data(opt.batch_size // queries, queries)
    batch_feature = torch.cat(batch_feature)
    batch_att = util.tensor_interleave(batch_att, queries)
    ##
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)

def WeightedL1(pred, gt):
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)
    
def generate_syn_feature(generator,classes, attribute,num,netF=None,netDec=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    support = []
    support_labels = []
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        syn_noisev = Variable(syn_noise,volatile=True)
        syn_attv = Variable(syn_att,volatile=True)
        fake = generator(syn_noisev,c=syn_attv)
        if netF is not None:
            dec_out = netDec(fake) # only to call the forward function of decoder
            dec_hidden_feat = netDec.getLayersOutDet() #no detach layers
            feedback_out = netF(dec_hidden_feat)
            fake = generator(syn_noisev, a1=opt.a2, c=syn_attv, feedback_layers=feedback_out)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)
        ## NEW: return data directly for FSL
        support.append(Variable(output.data))
        support_labels.append(iclass)

    return syn_feature, syn_label, support, support_labels


optimizer          = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerD         = optim.Adam(netD.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerG         = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerF         = optim.Adam(netF.parameters(), lr=opt.feed_lr, betas=(opt.beta1, 0.999))
optimizerDec       = optim.Adam(netDec.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
## NEW: classifier optimizer
optimizerClsf      = optim.Adam(clsf.parameters(), lr=opt.classifier_lr, betas=(opt.beta1, 0.999))
##


def calc_gradient_penalty(netD,real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

best_gzsl_acc = 0
best_zsl_acc = 0
for epoch in range(0,opt.nepoch):
    for loop in range(0,opt.feedback_loop):
        ## BEFORE
        # for i in range(0, data.ntrain, opt.batch_size):
        ## AFTER
        for i in range(0, len(data), opt.batch_size):
        ##
            ######### Discriminator training ##############
            for p in netD.parameters(): #unfreeze discriminator
                p.requires_grad = True

            for p in netDec.parameters(): # unfreeze decoder
                p.requires_grad = True
            # Train D1 and Decoder (and Decoder Discriminator)
            gp_sum = 0 #lAMBDA VARIABLE
            for iter_d in range(opt.critic_iter):
                sample()
                netD.zero_grad()
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)

                ## SE decoder training with real samples 
                netDec.zero_grad()
                recons = netDec(input_resv)
                R_cost = opt.recons_weight*WeightedL1(recons, input_attv) 
                R_cost.backward()
                optimizerDec.step()

                ## wgan training with real samples
                criticD_real = netD(input_resv, input_attv)
                criticD_real = opt.gammaD*criticD_real.mean()
                criticD_real.backward(mone)

                if opt.encoded_noise:  # <--- Use VAE
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                    eps = Variable(eps.cuda())
                    z = eps * std + means #torch.Size([64, 312])
                else:
                    noise.normal_(0, 1)
                    z = Variable(noise)

                ## generator (not trained) and feedback (also not trained)
                if loop >= 1:
                    fake = netG(z, c=input_attv)
                    dec_out = netDec(fake)
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    fake = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                else:
                    fake = netG(z, c=input_attv)

                # wgan training with fake samples
                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = opt.gammaD*criticD_fake.mean()
                criticD_fake.backward(one)
                # gradient penalty
                gradient_penalty = opt.gammaD*calc_gradient_penalty(netD, input_res, fake.data, input_att)
                # if opt.lambda_mult == 1.1:
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty #add Y here and #add vae reconstruction loss
                optimizerD.step()

            gp_sum /= (opt.gammaD*opt.lambda1*opt.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                opt.lambda1 /= 1.1

            ## NEW: classifier finetuning
            clsf.zero_grad()
            noise.normal_(0, 1)
            z = Variable(noise)

            if loop >= 1:
                recon_x = netG(z, c=input_attv)
                dec_out = netDec(recon_x)
                dec_hidden_feat = netDec.getLayersOutDet()
                feedback_out = netF(dec_hidden_feat)
                recon_x = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
            else:
                recon_x = netG(z, c=input_attv)

            support = [recon_x[i:i+shot] for i in range(0, recon_x.size(0), shot)]
            query = [input_resv[i:i+queries] for i in range(0, input_resv.size(0), queries)]
            logits = clsf(support, query)
            clsf_cost = clsf_loss(logits)
            clsf_cost.backward()
            optimizerClsf.step()
            ##

            ############# Generator training ##############
            sample()
            # Train Generator and Decoder
            for p in netD.parameters(): #freeze discriminator
                p.requires_grad = False
            if opt.recons_weight > 0 and opt.freeze_dec:
                for p in netDec.parameters(): #freeze decoder
                    p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()
            netF.zero_grad()
            clsf.zero_grad()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            # encoder
            means, log_var = netE(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
            eps = Variable(eps.cuda())
            z = eps * std + means #torch.Size([64, 312])
            # generator and feedback
            if loop >= 1:
                recon_x = netG(z, c=input_attv)
                dec_out = netDec(recon_x)
                dec_hidden_feat = netDec.getLayersOutDet()
                feedback_out = netF(dec_hidden_feat)
                recon_x = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
            else:
                recon_x = netG(z, c=input_attv)

            vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var) # minimize E 3 with this setting feedback will update the loss as well
            errG = vae_loss_seen

            if opt.encoded_noise:  # <--- Use VAE
                criticG_fake = netD(recon_x,input_attv).mean()
                fake = recon_x
            else:
                noise.normal_(0, 1)
                noisev = Variable(noise)
                if loop >= 1:
                    fake = netG(noisev, c=input_attv)
                    dec_out = netDec(recon_x) #Feedback from Decoder encoded output
                    dec_hidden_feat = netDec.getLayersOutDet()
                    feedback_out = netF(dec_hidden_feat)
                    fake = netG(noisev, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                else:
                    fake = netG(noisev, c=input_attv)
                criticG_fake = netD(fake,input_attv).mean()

            ## NEW: classifier loss backprop to gen etc
            noise.normal_(0, 1)
            z = Variable(noise)

            if loop >= 1:
                recon_x = netG(z, c=input_attv)
                dec_out = netDec(recon_x)
                dec_hidden_feat = netDec.getLayersOutDet()
                feedback_out = netF(dec_hidden_feat)
                recon_x = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
            else:
                recon_x = netG(z, c=input_attv)

            support = [recon_x[i:i+shot] for i in range(0, recon_x.size(0), shot)]
            query = [input_resv[i:i+queries] for i in range(0, input_resv.size(0), queries)]
            logits = clsf(support, query)
            clsf_cost = clsf_loss(logits)
            ##

            G_cost = -criticG_fake
            errG += opt.gammaG*G_cost
            netDec.zero_grad()
            recons_fake = netDec(fake)
            R_cost = WeightedL1(recons_fake, input_attv)
            errG += opt.recons_weight * R_cost
            errG += opt.clsf_weight * clsf_cost
            errG.backward()
            # write a condition here
            optimizer.step()  # encoder
            optimizerG.step()  # generator
            if loop >= 1:
                optimizerF.step()  # feedback
            if opt.recons_weight > 0 and not opt.freeze_dec: # not train decoder at feedback time
                optimizerDec.step()  # attdec

    print(
        ('[%d/%d]  Loss_D: %.4f, Loss_G: %.4f, '
         'Wasserstein_dist: %.4f, vae_loss_seen: %.4f, clsf loss: %.4f') %
        (
            epoch,
            opt.nepoch,
            D_cost.data[0],
            G_cost.data[0],
            Wasserstein_D.data[0],
            vae_loss_seen.data[0],
            clsf_cost.data[0]
        ),
        end=" "
    )
    netG.eval()
    netDec.eval()
    netF.eval()
    ## BEFORE
    # syn_feature, syn_label = generate_syn_feature(netG,data.unseenclasses, data.attribute, opt.syn_num,netF=netF,netDec=netDec)
    ## AFTER
    if opt.gzsl:
        syn_label = torch.LongTensor(np.unique(data.eval_unseen_labels))
    else:
        syn_label = torch.LongTensor(np.unique(data.eval_labels))
    syn_feature, syn_label, support, support_labels = generate_syn_feature(
        netG,
        syn_label,  # need dataset labels to get attributes
        data.attributes,
        opt.syn_num,
        netF=netF,
        netDec=netDec,
    )
    syn_label = torch.LongTensor([data.eval_label_mapping[label] for label in syn_label])
    support_labels = [data.eval_label_mapping[label] for label in support_labels]
    ##

    ## BEFORE
    # # Generalized zero-shot learning
    # if opt.gzsl:
    #     # Concatenate real seen features with synthesized unseen features
    #     train_X = torch.cat((data.train_feature, syn_feature), 0)
    #     train_Y = torch.cat((data.train_label, syn_label), 0)
    #     nclass = opt.nclass_all
    #     # Train GZSL classifier
    #     gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
    #             25, opt.syn_num, generalized=True, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
    #     if best_gzsl_acc < gzsl_cls.H:
    #         best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
    #     print('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H),end=" ")

    # # Zero-shot learning
    # # Train ZSL classifier
    # zsl_cls = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
    #                 data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, \
    #                 generalized=False, netDec=netDec, dec_size=opt.attSize, dec_hidden_size=4096)
    # acc = zsl_cls.acc
    # if best_zsl_acc < acc:
    #     best_zsl_acc = acc
    # print('ZSL: unseen accuracy=%.4f' % (acc))
    # # reset G to training mode
    # netG.train()
    # netDec.train()
    # netF.train()

# print('Dataset', opt.dataset)
# print('the best ZSL unseen accuracy is', best_zsl_acc)
# if opt.gzsl:
#     print('Dataset', opt.dataset)
#     print('the best GZSL seen accuracy is', best_acc_seen)
#     print('the best GZSL unseen accuracy is', best_acc_unseen)
#     print('the best GZSL H is', best_gzsl_acc)

    ## AFTER
    if opt.gzsl:
        # train_X = torch.cat((data.train_features, syn_feature))
        # train_labels = torch.tensor(
        #     [data.train_label_mapping[label] for label in data.train_labels]
        # )
        # train_Y = torch.cat((train_labels, syn_label))
        # nclass = data.n_classes
        # synthetic support right now
        acc, acc_s, acc_u = classifier.eval_protonet(clsf, data, support, support_labels)
        if acc > best_gzsl_acc:
            best_gzsl_acc, best_acc_seen, best_acc_unseen = acc, acc_s, acc_u
            best_epoch = epoch
        print('GZSL: s=%.4f, u=%.4f, H=%.4f' % (acc_s * 100, acc_u * 100, acc * 100))
    else:
        acc = classifier.eval_protonet(clsf, data, support, support_labels)
        if acc > best_zsl_acc:
            best_zsl_acc = acc
            best_epoch = epoch
        print('ZSL: unseen accuracy=%.4f' % (acc * 100))

    netG.train()
    netDec.train()
    netF.train()

if opt.gzsl:
    print(
        f'Best GZSL results in {data.benchmark.upper()}: Epoch={best_epoch},'
        ' H={best_gzsl_acc*100:.2f}% s={best_acc_seen*100:.2f}%'
        ' u={best_acc_unseen*100:.2f}'
    )
else:
    print(
        f'Best ZSL results in {data.benchmark.upper()}: Epoch={best_epoch},'
        ' Acc={best_zsl_acc*100:.2f}%'
    )
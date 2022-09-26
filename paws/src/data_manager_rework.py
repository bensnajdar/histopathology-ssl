# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import time

from logging import getLogger

import numpy as np
from math import ceil

import torch
import torch as t

import torchvision.transforms as transforms
import torchvision

from determined.pytorch import DataLoader

import PIL
from PIL import Image
from PIL import ImageFilter

import h5py


from typing import Callable, Optional
from cbmi_utils.pytorch.datasets.kather import Kather224x224
from cbmi_utils.pytorch.datasets.kather import Kather96x96
from cbmi_utils.pytorch.datasets.kather import Kather
from cbmi_utils.pytorch.datasets.pcam import PatchCamelyon
from cbmi_utils.pytorch.datasets.wsss4luad import WSSS4LUAD96x96
from cbmi_utils.pytorch.datasets.icpr import ICPR96x96Balanced, ICPR96x96Unbalanced
from cbmi_utils.pytorch.datasets.midog import Midog224x224
from cbmi_utils.pytorch.datasets.colossal import ColossalSet224x224
from cbmi_utils.pytorch.datasets.crush import Crush96x96
from cbmi_utils.pytorch.datasets.tcga import TCGA
from cbmi_utils.pytorch.datasets.lizard import LizardClassification

_GLOBAL_SEED = 0
logger = getLogger()


def make_labels_matrix(num_classes, s_batch_size, device, world_size=1, unique_classes=False, smoothing=0.0):
    """
    Make one-hot labels matrix for labeled samples

    NOTE: Assumes labeled data is loaded with ClassStratifiedSampler from
          src/data_manager.py
    """

    local_images = s_batch_size*num_classes
    total_images = local_images*world_size

    off_value = smoothing/(num_classes*world_size) if unique_classes else smoothing/num_classes

    if unique_classes:
        labels = torch.zeros(total_images, num_classes*world_size).to(device) + off_value
        for r in range(world_size):
            # -- index range for rank 'r' images
            s1 = r * local_images
            e1 = s1 + local_images
            # -- index offset for rank 'r' classes
            offset = r * num_classes
            for i in range(num_classes):
                labels[s1:e1][i::num_classes][:, offset+i] = 1. - smoothing + off_value
    else:
        labels = torch.zeros(total_images, num_classes*world_size).to(device) + off_value
        for i in range(num_classes):
            labels[i::num_classes][:, i] = 1. - smoothing + off_value

    return labels


    
def get_TransHisto(dataset):  
    
    DS_NOT_DOUND = 'not found -- only following sets are implemented Kather, PatchCamelyon, WSSS4LUAD, ICPR, ICPR-BAL, MIDOG'
    
    class TransHisto(dataset):

        def __init__(
            self,
            root,
            sub_set = 'train',
            split_size = 0.08,
            split_seed = 42,
            split_file = False,
            transform = None,
            transform_target = None,
            init_transform = None,
            supervised = True,
            multicrop_transform = (0, None),
            supervised_views = 1
        ):

            
            super().__init__(root, sub_set, transform, transform_target)

            self.supervised_views = supervised_views
            self.multicrop_transform = multicrop_transform
            self.supervised = supervised
            self.transform = transform
            
            
            if self.supervised:
                # self.targets, self.samples = init_transform(self.targets, self.samples)   
                # replace with get_split from cbmi_utils and loop 
                if split_file:                    
                    split_indices = []
                    with open(split_file, 'r') as rfile:
                        for line in rfile:
                            indx = int(line.split('\n')[0])
                            split_indices.append(indx)          
                elif split_size is not None:
                    supervised_temp_dataset = self.get_split(size = split_size, seed = split_seed)
                    split_indices = supervised_temp_dataset.indices
                else:
                    raise NotImplementedError('No split configuration given')
                    
                new_targets, new_samples = [], []
                for indx in split_indices:
                    new_targets.append(self.targets[indx])
                    new_samples.append(self.samples[indx])

                self.targets = np.array(new_targets)
                self.samples = np.array(new_samples)

                self.target_indices = []
                for t in range(len(self.classes)):
                    indices = np.squeeze(np.argwhere(self.targets == t)).tolist()
                    self.target_indices.append(indices)


        def __getitem__(self, index):


            img, target = self.samples[index] , self.targets[index]
            img = Image.fromarray(img)


            if self.transform_target is not None:
                target = self.transform_target(target)
            
            if self.transform is not None:
                if self.supervised:
                    
                    transform = self.transform
                    py37_hack = *[transform(img) for _ in range(self.supervised_views)], target
                    return py37_hack
                    
                else:
                    img_1 = self.transform(img)
                    img_2 = self.transform(img)
                    multicrop, mc_transform = self.multicrop_transform
                    if multicrop > 0 and mc_transform is not None:
                        mc_imgs = [mc_transform(img) for _ in range(int(multicrop))]
                        py37_hack = img_1, img_2, *mc_imgs, target
                        return py37_hack
                       
                    return img_1, img_2, target
            
            return img, target
        
    
    return TransHisto

    
def init_data(
    dataset_name,
    transform,
    u_batch_size,
    s_batch_size,
    classes_per_batch,
    split_size = 0.08,
    split_seed = 42,
    split_file = False,
    finetune = False,
    unique_classes=False,
    multicrop_transform=(0, None),
    supervised_views=1,
    world_size=1,
    rank=0,
    sub_set='train',
    stratify=False,
    drop_last=True
):
    """
    :param dataset_name: ['pcam']
    :param transform: torchvision transform to apply to each batch of data
    :param init_transform: transform to apply once to all data at the start
    :param u_batch_size: unsupervised batch-size
    :param s_batch_size: supervised batch-size (images per class)
    :param classes_per_batch: num. classes sampled in each supervised batch per gpu
    :param finetune: flag for loading dataset for downstream task
    :param unique_classes: whether each GPU should load different classes
    :param multicrop_transform: number of smaller multi-crop images to return
    :param supervised_views: number of views to generate of each labeled imgs
    :param world_size: number of workers for distributed training
    :param rank: rank of worker in distributed training
    :param root_path: path to the root directory containing all dataset
    :param sub_set: whether to load training data
    :param stratify: whether to class stratify 'fine_tune' data loaders
    """
    
    if not finetune:
        return _init_histo_data(
            dataset_name=dataset_name,
            split_size=split_size,
            split_seed=split_seed,
            split_file=split_file,
            transform=transform,
            u_batch_size=u_batch_size,
            s_batch_size=s_batch_size,
            classes_per_batch=classes_per_batch,
            multicrop_transform=multicrop_transform,
            supervised_views=supervised_views,
            world_size=world_size,
            rank=rank,
            sub_set=sub_set)
    
    elif finetune:
        batch_size = s_batch_size
        return _init_histo_ft_data(
            dataset_name,
            split_size=split_size,
            split_seed=split_seed,
            split_file=split_file,
            transform=transform,
            batch_size=batch_size,
            stratify=stratify,
            classes_per_batch=classes_per_batch,
            unique_classes=unique_classes,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
            sub_set=sub_set)
    

def _init_histo_data(
    dataset_name,
    split_size,
    split_seed,
    split_file,
    transform,
    u_batch_size,
    s_batch_size,
    classes_per_batch=2,
    supervised_transform=None,
    multicrop_transform=(0, None),
    supervised_views=1,
    world_size=1,
    rank=0,
    sub_set='train',
):
    
    if dataset_name == 'kather':
        TransHisto_Class = get_TransHisto(Kather96x96)
        root_path = '/data/ldap/histopathologic/processed_read_only/Kather_H5/res96'
        #root_path = '../../datasets/processed_read_only/Kather_H5/res224'
    elif dataset_name == 'kather_norm_224':
        TransHisto_Class = get_TransHisto(Kather)
        root_path = '/data/ldap/histopathologic/processed_read_only/kather/h5/224/norm_split0.9'
    elif dataset_name == 'crush96':
        TransHisto_Class = get_TransHisto(Crush96x96)
        root_path = '/data/ldap/histopathologic/processed_read_only/Crush_96/'
    elif dataset_name == 'patchcamelyon':
        TransHisto_Class = get_TransHisto(PatchCamelyon)
        root_path = '/data/ldap/histopathologic/original_read_only/PCam/PCam'
    elif dataset_name == 'lizard':
        TransHisto_Class = get_TransHisto(LizardClassification)
        root_path = '/data/ldap/histopathologic/processed_read_only/lizard_classification'
    elif dataset_name == 'wsss4luad':
        TransHisto_Class = get_TransHisto(WSSS4LUAD96x96)
        root_path = '/data/ldap/histopathologic/processed_read_only/WSSS4LUAD_96' 
    elif dataset_name == 'tcga':
        TransHisto_Class = get_TransHisto(TCGA)
        # this is necessary to use split directory for train and valid and parent directory for test
        if sub_set == 'test':
            root_path = '/data/ldap/histopathologic/processed_read_only/TCGA/TOAD/all'
        else:
            root_path = '/data/ldap/histopathologic/processed_read_only/TCGA/TOAD/all/split'
    else:
        raise NotImplementedError('dataset not found')
        
    
    unsupervised_set = TransHisto_Class(
        root=root_path,
        transform=transform,
        multicrop_transform=multicrop_transform,
        sub_set=sub_set,
        split_seed = split_seed,
        split_size = split_size,
        split_file = split_file,
        supervised=False)
    
    unsupervised_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=unsupervised_set,
        num_replicas=world_size,
        rank=rank)
    
    unsupervised_loader = DataLoader(
        unsupervised_set,
        sampler=unsupervised_sampler,
        batch_size=u_batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=8)

    supervised_sampler, supervised_loader = None, None
    if classes_per_batch > 0 and s_batch_size > 0:
        supervised_set = TransHisto_Class(root=root_path,
                                  transform=supervised_transform if supervised_transform is not None else transform,
                                  supervised_views=supervised_views,
                                  split_size = split_size,
                                  split_seed = split_seed,
                                  split_file = split_file,
                                  sub_set=sub_set,
                                  supervised=True)
        
        supervised_sampler = ClassStratifiedSampler(data_source=supervised_set,
                                                    world_size=world_size,
                                                    rank=rank,
                                                    batch_size=s_batch_size,
                                                    classes_per_batch=classes_per_batch,
                                                    seed=_GLOBAL_SEED)
        
        supervised_loader = DataLoader(supervised_set,
                                       batch_sampler=supervised_sampler,
                                       num_workers=8)
        
        if len(supervised_loader) > 0:
            tmp = ceil(len(unsupervised_loader) / len(supervised_loader))
            supervised_sampler.set_inner_epochs(tmp)
            logger.debug(f'supervised-reset-period {tmp}')

    return (unsupervised_loader, unsupervised_sampler,
            supervised_loader, supervised_sampler)


def _init_histo_ft_data(
    dataset_name,
    transform,
    batch_size,
    split_size,
    split_seed,
    split_file,
    sub_set,
    supervised_transform = None,
    stratify=False,
    classes_per_batch=1,
    unique_classes=False,
    supervised_views=1,
    world_size=1,
    rank=0,
    drop_last=False):
    
    if dataset_name == 'kather':
        TransHisto_Class = get_TransHisto(Kather96x96)
        root_path = '/data/ldap/histopathologic/processed_read_only/Kather_H5/res96'
        #root_path = '../../datasets/processed_read_only/Kather_H5/res96'
    elif dataset_name == 'kather_norm_224':
         TransHisto_Class = get_TransHisto(Kather)
         root_path = '/data/ldap/histopathologic/processed_read_only/kather/h5/224/norm_split0.9'
    elif dataset_name == 'crush96':
        TransHisto_Class = get_TransHisto(Crush96x96)
        root_path = '/data/ldap/histopathologic/processed_read_only/Crush_96/'
    elif dataset_name == 'patchcamelyon':
        TransHisto_Class = get_TransHisto(PatchCamelyon)
        root_path = '/data/ldap/histopathologic/original_read_only/PCam/PCam'
    elif dataset_name == 'lizard':
        TransHisto_Class = get_TransHisto(LizardClassification)
        root_path = '/data/ldap/histopathologic/processed_read_only/lizard_classification'
    elif dataset_name == 'wsss4luad':
        TransHisto_Class = get_TransHisto(WSSS4LUAD96x96)
        root_path = '/data/ldap/histopathologic/processed_read_only/WSSS4LUAD_96'
    elif dataset_name == 'tcga':
        TransHisto_Class = get_TransHisto(TCGA)
        # this is necessary to use split directory for train and valid and parent directory for test
        if sub_set == 'test':
            root_path = '/data/ldap/histopathologic/processed_read_only/TCGA/TOAD/all'
        else:
            root_path = '/data/ldap/histopathologic/processed_read_only/TCGA/TOAD/all/split'
    else:
        raise NotImplementedError('dataset not found')
        
    
    dataset = TransHisto_Class(root=root_path,
                              transform=supervised_transform if supervised_transform is not None else transform,
                              supervised_views = supervised_views,
                              split_seed = split_seed,
                              split_size = split_size,
                              split_file = split_file,
                              sub_set=sub_set,
                              supervised=True)

    if not stratify:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank)
        data_loader = DataLoader(
            dataset,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=True,
            num_workers=8)
    else:
        dist_sampler = ClassStratifiedSampler(
            data_source=dataset,
            world_size=world_size,
            rank=rank,
            batch_size=batch_size,
            classes_per_batch=classes_per_batch,
            seed=_GLOBAL_SEED,
            unique_classes=unique_classes)
        data_loader = DataLoader(
            dataset,
            batch_sampler=dist_sampler,
            pin_memory=True,
            num_workers=8)

    return (data_loader, dist_sampler)




def make_transforms(
    dataset_name,
    basic_augmentations=False,
    force_center_crop=False,
    std_resize_resolution = 32,
    crop_scale=(0.08, 1.0),
    color_jitter=1.0,
    normalize=False,
):
    """
    :param dataset_name: ['pcam']
    :param subset_path: path to .txt file denoting subset of data to use
    :param unlabeled_frac: fraction of data that is unlabeled
    :param training: whether to load training data
    :param basic_augmentations: whether to use simple data-augmentations
    :param force_center_crop: whether to force use of a center-crop
    :param color_jitter: strength of color-jitter
    :param normalize: whether to normalize color channels
    """

    return _make_histo_transforms(
        basic=basic_augmentations,
        force_center_crop=force_center_crop,
        normalize=normalize,
        color_distortion=color_jitter,
        std_resize_resolution = std_resize_resolution,
        scale=crop_scale
    )



def _make_histo_transforms(
    basic=False,
    force_center_crop=False,
    normalize=False,
    std_resize_resolution = 32, # size=32 paper standard for Cifar10 and Imagenet
    scale=(0.5, 1.0),
    color_distortion=0.5,
    keep_file=None,
    training='train'
):
    """
    Make data transformations

    :param unlabel_prob:probability of sampling unlabeled data point
    :param training: generate data transforms for train (alternativly test)
    :param basic: whether train transforms include more sofisticated transforms
    :param force_center_crop: whether to override settings and apply center crop to image
    :param normalize: whether to normalize image means and stds
    :param scale: random scaling range for image before resizing
    :param color_distortion: strength of color distortion
    :param keep_file: file containing names of images to use for semisupervised
    """
    
    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)

        def Solarize(img):
            v = np.random.uniform(0, 256)
            return PIL.ImageOps.solarize(img, v)
        solarize = transforms.Lambda(Solarize)
        rnd_solarize = transforms.RandomApply([solarize], p=0.2)

        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_solarize])
        return color_distort
    

    if training=='train' and (not force_center_crop):
        if basic:
            transform = transforms.Compose(
                [transforms.Resize(int(96)),
                 transforms.CenterCrop(size=std_resize_resolution), 
                 #transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()])
        else:
            transform = transforms.Compose(
                [transforms.Resize(int(96)),
                 transforms.Resize(int(std_resize_resolution*(2**0.5)+0.9999)),
                 transforms.RandomRotation(180),
                 transforms.CenterCrop(size=std_resize_resolution), #, scale=scale),
                 transforms.RandomHorizontalFlip(),
                 get_color_distortion(s=color_distortion),
                 transforms.ToTensor()]) 
    else:
        transform = transforms.Compose(
            [transforms.Resize(int(96)),
             transforms.CenterCrop(size=std_resize_resolution),
             transforms.ToTensor()])

    if normalize:
        transform = transforms.Compose(
            [transform,
             transforms.Normalize(
                 (0.7455422, 0.52883655, 0.70516384),
                 (0.15424888, 0.20247863, 0.14497302))]) 
    
    return transform



def make_multicrop_transform(
    dataset_name,
    num_crops,
    size,
    crop_scale,
    normalize,
    color_distortion
):
    
    return _make_multicrop_histo_transforms(
        num_crops=num_crops,
        size=size,
        scale=crop_scale,
        normalize=normalize,
        color_distortion=color_distortion)

def _make_multicrop_histo_transforms(
    num_crops,
    size=18,
    scale=(0.3, 0.75),
    normalize=False,
    color_distortion=0.5
):
    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)

        def Solarize(img):
            v = np.random.uniform(0, 256)
            return PIL.ImageOps.solarize(img, v)
        solarize = transforms.Lambda(Solarize)
        rnd_solarize = transforms.RandomApply([solarize], p=0.2)

        def Equalize(img):
            return PIL.ImageOps.equalize(img)
        equalize = transforms.Lambda(Equalize)
        rnd_equalize = transforms.RandomApply([equalize], p=0.2)

        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_solarize,
            rnd_equalize])
        return color_distort
    
    transform = transforms.Compose(
        [transforms.Resize(int(96)),
         transforms.Resize(int(96*(2**0.5)+0.9999)),
         transforms.RandomRotation(180),
         transforms.RandomResizedCrop(size=size, scale=scale),
         transforms.RandomHorizontalFlip(),
         get_color_distortion(s=color_distortion),
         transforms.ToTensor()])

    if normalize:
        transform = transforms.Compose(
            [transform,
             transforms.Normalize(
                 (0.7455422, 0.52883655, 0.70516384),
                 (0.15424888, 0.20247863, 0.14497302))])  #old normalize (0.65, 0.46, 0.65),(0.23, 0.23, 0.18)
        
        # normalize values for histo sets 
        # for tcga 
        #(0.7455422, 0.52883655, 0.70516384),
        #(0.15424888, 0.20247863, 0.14497302)
        
        # for pcam 
        #(0.7008, 0.5375, 0.6913),
        #(0.2168, 0.2611, 0.1937)
        
        # for lizard 
        #(0.64788544, 0.4870253,  0.68022424),
        #(0.2539682,  0.22869842, 0.24064516)
    return (num_crops, transform)


class ClassStratifiedSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        data_source,
        world_size,
        rank,
        batch_size=1,
        classes_per_batch=10,
        epochs=1,
        seed=0,
        unique_classes=False
    ):
        """
        ClassStratifiedSampler

        Batch-sampler that samples 'batch-size' images from subset of randomly
        chosen classes e.g., if classes a,b,c are randomly sampled,
        the sampler returns
            torch.cat([a,b,c], [a,b,c], ..., [a,b,c], dim=0)
        where a,b,c, are images from classes a,b,c respectively.
        Sampler, samples images WITH REPLACEMENT (i.e., not epoch-based)

        :param data_source: dataset of type "TransImageNet" or "TransCIFAR10'
        :param world_size: total number of workers in network
        :param rank: local rank in network
        :param batch_size: num. images to load from each class
        :param classes_per_batch: num. classes to randomly sample for batch
        :param epochs: num consecutive epochs thru data_source before gen.reset
        :param seed: common seed across workers for subsampling classes
        :param unique_classes: true ==> each worker samples a distinct set of classes; false ==> all workers sample the same classes
        """
        super(ClassStratifiedSampler, self).__init__(data_source)
        self.data_source = data_source

        self.rank = rank
        self.world_size = world_size
        self.cpb = classes_per_batch
        self.unique_cpb = unique_classes
        self.batch_size = batch_size
        self.num_classes = len(data_source.classes)
        self.epochs = epochs
        self.outer_epoch = 0

        if not self.unique_cpb:
            assert self.num_classes % self.cpb == 0

        self.base_seed = seed  # instance seed
        self.seed = seed  # subsample sampler seed

    def set_epoch(self, epoch):
        self.outer_epoch = epoch

    def set_inner_epochs(self, epochs):
        self.epochs = epochs

    def _next_perm(self):
        self.seed += 1
        g = torch.Generator()
        g.manual_seed(self.seed)
        self._perm = torch.randperm(self.num_classes, generator=g)

    def _get_perm_ssi(self):
        start = self._ssi
        end = self._ssi + self.cpb
        subsample = self._perm[start:end]
        return subsample

    def _next_ssi(self):
        if not self.unique_cpb:
            self._ssi = (self._ssi + self.cpb) % self.num_classes
            if self._ssi == 0:
                self._next_perm()
        else:
            self._ssi += self.cpb * self.world_size
            max_end = self._ssi + self.cpb * (self.world_size - self.rank)
            if max_end > self.num_classes:
                self._ssi = self.rank * self.cpb
                self._next_perm()

    def _get_local_samplers(self, epoch):
        """ Generate samplers for local data set in given epoch """
        seed = int(self.base_seed + epoch
                   + self.epochs * self.rank
                   + self.outer_epoch * self.epochs * self.world_size)
        g = torch.Generator()
        g.manual_seed(seed)
        samplers = []
        for t in range(self.num_classes):
            t_indices = np.array(self.data_source.target_indices[t])
            if not self.unique_cpb:
                i_size = len(t_indices) // self.world_size
                if i_size > 0:
                    t_indices = t_indices[self.rank*i_size:(self.rank+1)*i_size]
            if len(t_indices) > 1:
                t_indices = t_indices[torch.randperm(len(t_indices), generator=g)]
            samplers.append(iter(t_indices))
        return samplers

    def _subsample_samplers(self, samplers):
        """ Subsample a small set of samplers from all class-samplers """
        subsample = self._get_perm_ssi()
        subsampled_samplers = []
        for i in subsample:
            subsampled_samplers.append(samplers[i])
        self._next_ssi()
        return zip(*subsampled_samplers)

    def __iter__(self):
        self._ssi = self.rank*self.cpb if self.unique_cpb else 0
        self._next_perm()

        # -- iterations per epoch (extract batch-size samples from each class)
        ipe = (self.num_classes // self.cpb if not self.unique_cpb
               else self.num_classes // (self.cpb * self.world_size)) * self.batch_size

        for epoch in range(self.epochs):

            # -- shuffle class order
            samplers = self._get_local_samplers(epoch)
            subsampled_samplers = self._subsample_samplers(samplers)

            counter, batch = 0, []
            for i in range(ipe):
                batch += list(next(subsampled_samplers))
                counter += 1
                if counter == self.batch_size:
                    yield batch
                    counter, batch = 0, []
                    if i + 1 < ipe:
                        subsampled_samplers = self._subsample_samplers(samplers)

    def __len__(self):
        if self.batch_size == 0:
            return 0

        ipe = (self.num_classes // self.cpb if not self.unique_cpb
               else self.num_classes // (self.cpb * self.world_size))
        return self.epochs * ipe



class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

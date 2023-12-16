import os
import importlib

from .bmusimsiam import BmuSimSiam
from .bmubarlowtwins import BmuBarlowTwins
from .simsiam import SimSiam
from .barlowtwins import BarlowTwins
from .byol import BYOL
import torch
from .backbones import resnet18
from .moco import MoCo

def get_backbone(backbone, dataset, castrate=False):
    backbone = eval(f"{backbone}()")
    if dataset == 'seq-cifar100':
        backbone.n_classes = 100
    elif dataset == 'seq-cifar10':
        backbone.n_classes = 10
    backbone.output_dim = backbone.fc.in_features
    if not castrate:
        backbone.fc = torch.nn.Identity()

    return backbone


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]

def get_model(args, device, len_train_loader, transform):
    loss = torch.nn.CrossEntropyLoss()
    if args.model.name == 'simsiam':
        backbone =  SimSiam(get_backbone(args.model.backbone, args.dataset.name, args.cl_default)).to(device)
        if args.model.proj_layers is not None:
            backbone.projector.set_layers(args.model.proj_layers)
    elif args.model.name == 'barlowtwins':
        backbone = BarlowTwins(get_backbone(args.model.backbone, args.dataset.name, args.cl_default), device).to(device)
        if args.model.proj_layers is not None:
            backbone.projector.set_layers(args.model.proj_layers)
    elif args.model.name == 'bmusimsiam':
        backbone = BmuSimSiam(get_backbone(args.model.backbone, args.dataset.name, args.cl_default),get_backbone(args.model.backbone, args.dataset.name, args.cl_default)).to(device)
        if args.model.proj_layers is not None: #
            backbone.projector.set_layers(args.model.proj_layers)
            backbone.projector2.set_layers(args.model.proj_layers)
    elif args.model.name == 'bmubarlowtwins':
        backbone = BmuBarlowTwins(get_backbone(args.model.backbone, args.dataset.name, args.cl_default),get_backbone(args.model.backbone, args.dataset.name, args.cl_default),device).to(device)
        if args.model.proj_layers is not None:
            backbone.projector.set_layers(args.model.proj_layers)
            backbone.projector2.set_layers(args.model.proj_layers)

    names = {}
    for model in get_all_models():
        mod = importlib.import_module('models.' + model)
        class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
        names[model] = getattr(mod, class_name)
    
    return names[args.model.cl_model](backbone, loss, args, len_train_loader, transform)


import os
import math
import torch
import torch.nn as nn
from model.common import DownBlock
import model.drn


def dataparallel(model, gpu_list):
    ngpus = len(gpu_list)
    assert ngpus != 0, "only support gpu mode"
    assert torch.cuda.device_count() >= ngpus, "Invalid Number of GPUs"
    assert isinstance(model, list), "Invalid Type of Dual model"
    for i in range(len(model)):
        if ngpus >= 2:
            model[i] = nn.DataParallel(model[i], gpu_list).cuda()
        else:
            model[i] = model[i].cuda()
    return model


class Model(nn.Module):
    def __init__(self, opt, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.opt = opt
        self.scale = opt.scale
        self.idx_scale = 0
        self.self_ensemble = opt.self_ensemble
        self.cpu = opt.cpu
        self.device = torch.device('cpu' if opt.cpu else 'cuda')
        self.n_GPUs = opt.n_GPUs

        self.model = drn.make_model(opt).to(self.device)
        self.dual_models = []
        for _ in self.opt.scale:
            dual_model = DownBlock(opt, 2).to(self.device)
            self.dual_models.append(dual_model)
        
        if not opt.cpu and opt.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(opt.n_GPUs))
            self.dual_models = dataparallel(self.dual_models, range(opt.n_GPUs))

        self.load(opt.pre_train, opt.pre_train_dual, cpu=opt.cpu)

        if not opt.test_only:
            print(self.model, file=ckp.log_file)
            print(self.dual_models, file=ckp.log_file)
        
        # compute parameter
        num_parameter = self.count_parameters(self.model)
        ckp.write_log(f"The number of parameters is {num_parameter / 1000 ** 2:.2f}M")

    def forward(self, x, idx_scale=0):
        self.idx_scale = idx_scale
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)
        return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module
    
    def get_dual_model(self, idx):
        if self.n_GPUs == 1:
            return self.dual_models[idx]
        else:
            return self.dual_models[idx].module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)
    
    def count_parameters(self, model):
        if self.opt.n_GPUs > 1:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save(self, path, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(path, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(path, 'model', 'model_best.pt')
            )
        #### save dual models ####
        dual_models = []
        for i in range(len(self.dual_models)):
            dual_models.append(self.get_dual_model(i).state_dict())
        torch.save(
            dual_models,
            os.path.join(path, 'model', 'dual_model_latest.pt')
        )
        if is_best:
            torch.save(
                dual_models,
                os.path.join(path, 'model', 'dual_model_best.pt')
            )

    def load(self, pre_train='.', pre_train_dual='.', cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        #### load primal model ####
        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs),
                strict=False
            )
        #### load dual model ####
        if pre_train_dual != '.':
            print('Loading dual model from {}'.format(pre_train_dual))
            dual_models = torch.load(pre_train_dual, **kwargs)
            for i in range(len(self.dual_models)):
                self.get_dual_model(i).load_state_dict(
                    dual_models[i], strict=False
                )

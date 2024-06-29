from models.convnextv2 import ConvNeXtV2
import torch.nn as nn
from torch.nn import init
# from timm.models.layers import trunc_normal_
import torch

from models.model_utils import remap_checkpoint_keys, load_state_dict


def attr_mlp(input_dim, inter_dim, output_dim, after_cross, dropout_p):
    if after_cross:
        new_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(inter_dim, output_dim)
        )
    else:
        new_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, inter_dim),
            nn.BatchNorm1d(inter_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(inter_dim, output_dim)
        )
    init.normal_(new_mlp[1].weight.data, std=0.00001)
    init.constant_(new_mlp[1].bias.data, 0.0)
    init.normal_(new_mlp[4].weight.data, std=0.00001)
    init.constant_(new_mlp[4].bias.data, 0.0)
    # new_mlp[2].bias.requires_grad_(False) # no shift
    return new_mlp

class MultiConvnext2(nn.Module):
    def __init__(self, class_list, pre_train=True):
        # super().__init__(in_chans, num_classes, depths, dims, drop_path_rate, head_init_scale)
        super().__init__()
        self.base = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=1000, drop_path_rate=0.2, head_init_scale=0.001)
        self.fc_list = nn.ModuleList()
        self.projector_list = nn.ModuleList()
        embed_dim = 512
        for idx, class_num in enumerate(class_list):
            pj = nn.Sequential(
                nn.Linear(1024, embed_dim),
                nn.ReLU()
            )
            classifier = attr_mlp(embed_dim,(embed_dim//2),class_num,False,0.2)
            self.projector_list.append(pj)
            self.fc_list.append(classifier)
        # self.load_pretrain(load_path='/home/wuzesen/attributeReID/checkpoints/convnextv2_base_1k_224_fcmae.pt')
        if pre_train:
            self.load_pretrain(load_path='/home/wuzesen/attributeReID/checkpoints/convnextv2_base_22k_224_ema.pt')
        # /home/wuzesen/attributeReID/checkpoints/convnextv2_base_22k_224_ema.pt
        
        
        
    def load_pretrain(self, load_path):
        checkpoint = torch.load(load_path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % load_path)
        checkpoint_model = checkpoint['model']
        state_dict = self.base.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # remove decoder weights
        checkpoint_model_keys = list(checkpoint_model.keys())
        for k in checkpoint_model_keys:
            if 'decoder' in k or 'mask_token'in k or \
               'proj' in k or 'pred' in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        checkpoint_model = remap_checkpoint_keys(checkpoint_model)
        load_state_dict(self.base, checkpoint_model, prefix='')

        # manually initialize fc layer
        # trunc_normal_(self.base.head.weight, std=2e-5)
        # torch.nn.init.constant_(self.base.head.bias, 0.)

    def forward(self, image):
        image_feat_list = []
        pred_list = []
        x = self.base.forward_features(image)
        for idx, pj in enumerate(self.projector_list):
            attr_feat = pj(x)
            image_feat_list.append(attr_feat)
        for idx, fc in enumerate(self.fc_list):
            # pred_list.append(fc(x))
            pred_list.append(fc(image_feat_list[idx]))

        return pred_list

class MultiConvnext2Large(nn.Module):
    def __init__(self, class_list, pre_train=True):
        # super().__init__(in_chans, num_classes, depths, dims, drop_path_rate, head_init_scale)
        super().__init__()
        self.base = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], num_classes=1000, drop_path_rate=0.2, head_init_scale=0.001)
        self.fc_list = nn.ModuleList()
        self.projector_list = nn.ModuleList()
        embed_dim = 768
        for idx, class_num in enumerate(class_list):
            pj = nn.Sequential(
                nn.Linear(1536, embed_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                # nn.PReLU(),
                # nn.GELU(),
            )
            classifier = attr_mlp(embed_dim,(embed_dim//2),class_num,False,0.2)
            self.projector_list.append(pj)
            self.fc_list.append(classifier)
        if pre_train:
            self.load_pretrain(load_path='/home/wuzesen/attributeReID/checkpoints/convnextv2_large_22k_224_ema.pt')
        
        
        
    def load_pretrain(self, load_path):
        checkpoint = torch.load(load_path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % load_path)
        checkpoint_model = checkpoint['model']
        state_dict = self.base.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # remove decoder weights
        checkpoint_model_keys = list(checkpoint_model.keys())
        for k in checkpoint_model_keys:
            if 'decoder' in k or 'mask_token'in k or \
               'proj' in k or 'pred' in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        checkpoint_model = remap_checkpoint_keys(checkpoint_model)
        load_state_dict(self.base, checkpoint_model, prefix='')

        # manually initialize fc layer
        # trunc_normal_(self.base.head.weight, std=2e-5)
        # torch.nn.init.constant_(self.base.head.bias, 0.)

    def forward(self, image):
        image_feat_list = []
        pred_list = []
        x = self.base.forward_features(image)
        for idx, pj in enumerate(self.projector_list):
            attr_feat = pj(x)
            image_feat_list.append(attr_feat)
        for idx, fc in enumerate(self.fc_list):
            # pred_list.append(fc(x))
            pred_list.append(fc(image_feat_list[idx]))

        return pred_list
    
class MultiConvnext2Tiny(nn.Module):
    def __init__(self, class_list):
        # super().__init__(in_chans, num_classes, depths, dims, drop_path_rate, head_init_scale)
        super().__init__()
        self.base = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=1000, drop_path_rate=0.2, head_init_scale=0.001)
        self.fc_list = nn.ModuleList()
        self.projector_list = nn.ModuleList()
        embed_dim = 384
        for idx, class_num in enumerate(class_list):
            pj = nn.Sequential(
                nn.Linear(768, embed_dim),
                nn.ReLU()
            )
            # pj init
            # init.normal_(pj[0].weight.data, std=0.00001)
            # init.constant_(pj[0].bias.data, 0.0)

            classifier = attr_mlp(embed_dim,(embed_dim//2),class_num,False,0.2)
            self.projector_list.append(pj)
            self.fc_list.append(classifier)
        self.load_pretrain(load_path='/home/wuzesen/attributeReID/checkpoints/convnextv2_tiny_22k_224_ema.pt')
        
        
        
    def load_pretrain(self, load_path):
        checkpoint = torch.load(load_path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % load_path)
        checkpoint_model = checkpoint['model']
        state_dict = self.base.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # remove decoder weights
        checkpoint_model_keys = list(checkpoint_model.keys())
        for k in checkpoint_model_keys:
            if 'decoder' in k or 'mask_token'in k or \
               'proj' in k or 'pred' in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        checkpoint_model = remap_checkpoint_keys(checkpoint_model)
        load_state_dict(self.base, checkpoint_model, prefix='')

        # manually initialize fc layer
        # trunc_normal_(self.base.head.weight, std=2e-5)
        # torch.nn.init.constant_(self.base.head.bias, 0.)

    def forward(self, image):
        image_feat_list = []
        pred_list = []
        x = self.base.forward_features(image)
        for idx, pj in enumerate(self.projector_list):
            attr_feat = pj(x)
            image_feat_list.append(attr_feat)
        for idx, fc in enumerate(self.fc_list):
            # pred_list.append(fc(x))
            pred_list.append(fc(image_feat_list[idx]))

        return pred_list
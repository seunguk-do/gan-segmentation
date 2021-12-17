import os
import torch

from generator import Generator
from stylegan2_ada_pytorch import torch_utils


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.abspath(__file__))
    pretrained_model_path = os.path.join(root_path,
                                         'pretrained_models',
                                         'icgan_stylegan2_coco_res256',
                                         'test.pth')

    generator = Generator(z_dim=512, c_dim=0, h_dim=2048, w_dim=512,
                          img_resolution=256, img_channels=3,
                          mapping_kwargs={'num_layers':2},
                          synthesis_kwargs={'channel_base': 16384, 'channel_max': 512, 
                                            'num_fp16_res': 4, 'conv_clamp': 256})
    print("Loading")

    generator.load_state_dict(torch.load(pretrained_model_path))
    print("Load Complete")

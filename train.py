import os
import torch
import torch.nn.functional as F
import numpy as np
import sklearn
import time
import random
from stylegan2_ada_pytorch import torch_utils
from generator import Generator
from model import FewShotCNN
from ic_gan.data_utils.utils import load_pretrained_feature_extractor

# Environment Variables
root_path = os.path.dirname(os.path.abspath(__file__))

def load_feature_extractor_and_precomputed_features(path_to_swav, path_to_precomputed_features):
    feature_extractor = load_pretrained_feature_extractor(path_to_swav, "selfsupervised").eval()
    precomputed_features = np.load(path_to_precomputed_features, allow_pickle=True).item()["instance_features"]    
    precomputed_features = torch.tensor(precomputed_features, requires_grad=False, device="cpu")
    return feature_extractor, precomputed_features

def get_h(img, feature_extractor, precomputed_features):
    all_dists = []

    # get the feature of given img 
    with torch.no_grad():
        out_features, _ = feature_extractor(img)
    out_features = (out_features / torch.linalg.norm(out_features, dim=-1, keepdims=True)).cpu()

    # find the most similar k nn center given the feature of input img
    for i in range(len(precomputed_features)):
        dist = sklearn.metrics.pairwise_distances(
                out_features, precomputed_features[i], metric="euclidean", n_jobs=-1)
        all_dists.add(np.diagonal(dist))
    h_idx = np.argsort(all_dists)[0]

    return precomputed_features[h_idx].cuda() 

def get_ws(Gen, img, h):
    proj_training_step = 1000



    return ws



if __name__ == '__main__':
    # Training Arguments
    device = 'cuda:0' 
    image_size = 256
    n_samples = 
    pretrained_model_path = os.path.join(root_path, 'pretrained_models',
                                         'icgan_stylegan2_coco_res256',
                                         'best_model.pth')
    checkpoint_dir = os.path.join(root_path, 'checkpoint')
    path_to_swav = os.path.join(root_path, 'datasets', 'swav_800ep_pretrain.pth.tar')
    path_to_precomputed_features = os.path.join(root_path, 'pretrained_models', 'stored_instances', 'coco_res256_rn50_selfsupervised_kmeans_k1000_instance_features.npy')


    # Load pretrained features and SwaV feature extractor
    feature_extractor, precomputed_features = load_feature_extractor_and_precomputed_features(path_to_swav, path_to_precomputed_features)
    

    # Construct Networks
    generator = Generator(z_dim=512, c_dim=0, h_dim=2048, w_dim=512,
                          img_resolution=256, img_channels=3,
                          mapping_kwargs={'num_layers':2},
                          synthesis_kwargs={'channel_base': 16384, 
                                            'channel_max': 512, 
                                            'num_fp16_res': 4, 
                                            'conv_clamp': 256})
    generator.load_state_dict(torch.load(pretrained_model_path))

    model = FewShotCNN(in_ch, n_class)

    # Data Loader
    dataloader = 

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    model.train().to(device)
    start_time = time.time()
    for epoch in range(1, 100+1):
        sample_order = list(range(n_sample))
        random.shuffle(sample_order)

        for idx in sample_order:
            ## loader brings in img
            img, label = 

            h = get_h(img, feature_extractor, precomputed_features)
            ws = get_ws(generator, img, h)

            
            _, feat = generator(ws) 

            out = model(feat)

            loss = F.cross_entropy(out, label, reduction='mean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print(f'{epoch:5}-th epoch | loss: {loss.item():6.4f} | time: {time.time()-start_time:6.1f}sec')
            checkpoint_path = os.path.join(checkpoint_dir, f'val_loss_{loss.item():6.4f}.pt')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, checkpoint_path)

        scheduler.step()
    print('Done!')

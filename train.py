import os
import torch
import torch.nn.functional as F
import numpy as np
import sklearn.metrics
import time
import random
from stylegan2_ada_pytorch import torch_utils
from generator import Generator
from model import FewShotCNN
from ic_gan.data_utils.utils import load_pretrained_feature_extractor
from stylegan2_ada_pytorch import dnnlib
from data_utils import ImageDataset


# Environment Variables
root_path = os.path.dirname(os.path.abspath(__file__))

def concat_features(features):
    h = max([f.shape[-2] for f in features])
    w = max([f.shape[-1] for f in features])
    return torch.cat([torch.nn.functional.interpolate(f, (h,w), mode='nearest') for f in features], dim=1)

def preprocess_img(img):
    
    return img


def load_feature_extractor_and_precomputed_features(path_to_swav, path_to_precomputed_features):
    feature_extractor = load_pretrained_feature_extractor(path_to_swav, "selfsupervised").eval()
    precomputed_features = np.load(path_to_precomputed_features, allow_pickle=True).item()["instance_features"]    
    precomputed_features = torch.tensor(precomputed_features, requires_grad=False, device="cpu")
    return feature_extractor, precomputed_features

def get_h(img, feature_extractor, precomputed_features):
    # SwaV trained on 224
    img = torch.nn.functional.interpolate(img, 224, mode="bicubic", align_corners=True)
    all_dists = []

    # get the feature of given img 
    with torch.no_grad():
        out_features, _ = feature_extractor(img)
        out_features = (out_features / torch.linalg.norm(out_features, dim=-1, keepdims=True)).cpu()

    # find the most similar k nn center given the feature of input img
    for i in range(len(precomputed_features)):
        dist = sklearn.metrics.pairwise_distances(
                out_features, precomputed_features[i].unsqueeze(0), metric="euclidean", n_jobs=1)
        all_dists.append(np.diagonal(dist)[0])
    h_idx = np.argsort(all_dists)[0]

    return precomputed_features[h_idx].cuda() 

def get_ws(G, target, h, device):
    num_steps = 1000
    initial_learning_rate=0.1
    initial_noise_factor=0.05
    lr_rampdown_length=0.25
    lr_rampup_length=0.05
    noise_ramp_length=0.75
    regularize_noise_weight=1e5
    w_avg_samples = 10000

    h = h.repeat(w_avg_samples,1)
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None, h)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    noise_bufs = {
        name: buf
        for (name, buf) in G.synthesis.named_buffers()
        if "noise_const" in name
    }

        # Load VGG16 feature detector.
    url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode="area")
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(
        w_avg, dtype=torch.float32, device=device, requires_grad=True
    )  # pylint: disable=not-callable
    w_out = torch.zeros(
        [num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device
    )
    optimizer = torch.optim.Adam(
        [w_opt] + list(noise_bufs.values()),
        betas=(0.9, 0.999),
        lr=initial_learning_rate,
    )

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = (
            w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        )
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images, _ = G.synthesis(ws, noise_mode="const")

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode="area")

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}")

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out[num_steps-1].repeat([1, G.mapping.num_ws, 1])

if __name__ == '__main__':
    # Training Arguments
    device = 'cuda' 
    image_size = 256
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
                                            'conv_clamp': 256}).to(device)
    generator.load_state_dict(torch.load(pretrained_model_path))
    generator.requires_grad_(False)

    # Data Loader
    dataset = ImageDataset('./datasets/coco') 
    categories = dataset.get_category_ids()

    model = FewShotCNN(4416, 90, size='L')

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    model.train().to(device)
    start_time = time.time()
    for epoch in range(1, 100+1):
        
        random.shuffle(categories)

        for cat_id in categories:
            ## loader brings in img
            img, label = dataset.get_sample_by_category_id(cat_id)

            # img [N, C, W, H]
            h = get_h(img, feature_extractor, precomputed_features) #20~30sec
            ws = get_ws(generator, img, h, device)  #90 sec

            _, feat = generator(ws)

            feat_concat = concat_features(feat)
            out = model(feat_concat) # --> [1, n_class, 256, 256] 

            loss = F.cross_entropy(out, label, reduction='mean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print(f'{epoch:5}-th epoch | loss: {loss.item():6.4f} | time: {time.time()-start_time:6.1f}sec')
            checkpoint_path = os.path.join(checkpoint_dir, f'val_loss_{loss.item():6.4f}.pt')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, checkpoint_path)

        scheduler.step()
    print('Done!')

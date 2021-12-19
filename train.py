import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import random
from generator import Generator
from model import FewShotCNN
import pickle
import sys
from eval_utils import mIoUEstimator
from PIL import Image
from torchvision.utils import save_image

def save_as_img(target, name, normalized=True, mask=False):
    if mask == True:
        image = target.permute(1,2,0).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image.squeeze(axis=2))
        image.save(name)
    elif normalized == False:
        image = target.permute(0, 2, 3, 1).clamp(0,255).to(torch.uint8)[0].cpu().numpy()
        image = Image.fromarray(image)
        image.save(name)
    else:
        save_image(target, name)

# Environment Variables
root_path = os.path.dirname(os.path.abspath(__file__))

def concat_features(features):
    h = max([f.shape[-2] for f in features])
    w = max([f.shape[-1] for f in features])
    return torch.cat([torch.nn.functional.interpolate(f, (h,w), mode='nearest') for f in features], dim=1)

if __name__ == '__main__':
    num_train_set = int(sys.argv[1])  # amount of sets to use in training
    num_eval_set = int(sys.argv[2]) # amount of sets to use in evaluation

    assert num_train_set + num_eval_set <= 10

    # Training Arguments
    device = 'cuda' 
    image_size = 256
    pretrained_model_path = os.path.join(root_path, 'pretrained_models',
                                         'icgan_stylegan2_coco_res256',
                                         'best_model.pth')
    checkpoint_dir = os.path.join(root_path, 'checkpoints')

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

    # Load data
    with open("pickles_new/h_sets.pickle", "rb") as f:
        h_sets = pickle.load(f)
    with open("pickles_new/ws_sets.pickle", "rb") as f:
        ws_sets = pickle.load(f)
    with open("pickles_new/masks.pickle", "rb") as f:
        labels = pickle.load(f)

    train_h_set = []
    train_ws_set = []
    train_labels = []
    eval_h_set = []
    eval_ws_set = []
    eval_labels = []

    for i in range(num_train_set):
        train_h_set += h_sets[i]
        train_ws_set += ws_sets[i]
        train_labels += labels[i]
    for i in range(10 - num_eval_set, 10):
        eval_h_set += h_sets[i]
        eval_ws_set += ws_sets[i]
        eval_labels += labels[i]

    model = FewShotCNN(4416, 91, size='L')
    mIoU_estimator = mIoUEstimator()

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    model.train().to(device)
    start_time = time.time()
    print("Start training...!")
    for epoch in range(1, 100+1):
        print(f"epoch: {epoch}/100")
        for ws, label in zip(train_ws_set, train_labels):
            ws, label = ws.cuda(), label.cuda()
            with torch.no_grad():
                _,feat = generator(ws)
            feat_concat = concat_features(feat)
            out = model(feat_concat)
            loss = F.cross_entropy(out, label, reduction='mean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 5 == 0:
            model.eval()
            print(f'{epoch:5}-th epoch | loss: {loss.item():6.4f} | time: {time.time()-start_time:6.1f}sec')
            scores = []
            for ws, label in zip(eval_ws_set, eval_labels):
                ws, label = ws.cuda(), label.cuda()
                with torch.no_grad():
                    _, feat = generator(ws)
                    feat_concat = concat_features(feat)
                    out = model(feat_concat)
                    model_prediction = torch.max(out, dim=1)[1]
                # model_prediction & ground_truth : tensor of [B, H, W] or [B, 1, H, W]
                metric = mIoU_estimator(model_prediction, label)
                scores.append(metric.item())
            
            avg_score = np.mean(scores)
            print(f'average mIoU: {avg_score}')
            checkpoint_path = os.path.join(checkpoint_dir, f'mIoU_{avg_score:6.4f}.pt')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, checkpoint_path)
            model.train()

        scheduler.step()
    print('Done!')

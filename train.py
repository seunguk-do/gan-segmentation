import torch
from stylegan2_ada_pytorch import torch_utils
from generator import Generator
from model import FewShotCNN


root_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # Environment Variables
    device = 'cuda:0' 
    image_size = 256
    n_samples = 
    pretrained_model_path = os.path.join(root_path, 'pretrained_models',
                                         'icgan_stylegan2_coco_res256',
                                         'best_model.pth')
    checkpoint_dir = os.path.join(root_path, 'checkpoint')

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
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    model.train().to(device)
    start_time = time.time()
    for epoch in range(1, 100+1):
        sample_order = list(range(n_sample))
        random.shuffle(sample_order)

        for idx in sample_order:
            ws = get_ws()
            img, feat = generator(ws) 
            label = 

            out = model(sample)

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

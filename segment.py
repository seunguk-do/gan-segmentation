

def get_z(Gen, img, h):
    proj_training_step = 1000





def get_h(img, feature_extractor, precomp_features):
    all_dists = []

    # get the feature of given img 
    with torch.no_grad():
        out_features, _ = feature_extractor(img)
    out_features /= torch.linalg.norm(out_features,dim=-1, keepdims=True).cpu()

    # find the most similar k nn center given the feature of input img
    for i in len(precomp_features):
        dist = sklearn.metrics.pairwise_distances(
                out_features, precomp_features[i], metric="euclidean", n_jobs=-1)
        all_dists.add(np.diagonal(dist))
    h_idx = np.argsort(all_dists)[0]

    return precomp_features[h_idx].cuda()





def load_feature_extractor_and_precomputed_features(path_to_swav, path_to_pre_computed_features):
    feature_extractor = ic_gan.data_utils.load_pretrained_feature_extractor(path_to_swav, "selfsupervised").eval()
    precomp_features = np.load(path_to_pre_computed_features, allow_pickle=True).item()["instance_features"]    
    precomp_features = torch.tensor(precomp_features, requires_grad=False, device="cpu")
    return feature_extractor, precomp_features
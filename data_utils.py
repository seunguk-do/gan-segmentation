from pycocotools.coco import COCO 
import numpy as np 
import skimage.io as io 
import torchvision.transforms as transforms
import torch
#from torch.utils.data import Dataset

NORM_MEAN = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
NORM_STD = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Parameters
    ----------
        size: sequence or int
            Desired output size of the crop. If size is an int instead of sequence like (h, w),
            a square crop (size, size) is made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class ImageDataset:
    def __init__(self, data_dir, data_type='train2017'):
        ann_path = '{}/annotations/instances_{}.json'.format(data_dir,data_type)
        self.coco=COCO(ann_path)

    def get_category_names(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        names=[cat['name'] for cat in cats] 
        return names

    def get_category_ids(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        ids=[cat['id'] for cat in cats] 
        return ids
    
    def get_sample_by_category_id(self, cat_ids):
        """
        Args:
            cat_ids: [int]
                list of category ids to be included in the image
        Returns:
            image: np array [H, W, C]
            labels: lisf of segmentations
        """
        img_ids = self.coco.getImgIds(catIds=cat_ids)
        img = self.coco.loadImgs(img_ids[np.random.randint(0,len(img_ids))])[0]
        I = io.imread(img['coco_url'])

        ann_ids = self.coco.getAnnIds(imgIds=img['id'], catIds=self.coco.getCatIds(catIds=self.get_category_ids()), iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        # 1 for segmented area, 0 for others
        mask = self.coco.annToMask(anns[0]) * anns[0]['category_id']
        for i in range(1, len(anns)):
            mask += self.coco.annToMask(anns[i]) * anns[1]['category_id']
        
        I = transforms.ToTensor()(I).to(torch.float32).cuda().unsqueeze(0)
        mask = torch.from_numpy(mask).to(torch.float32).unsqueeze(0).cuda().unsqueeze(0)

        crop_dim = min(I.shape[2], I.shape[3])
        I, mask = transforms.CenterCrop(crop_dim)(I), transforms.CenterCrop(crop_dim)(mask)
        I, mask = transforms.Resize((256, 256))(I), transforms.Resize((256, 256))(mask)
        I = transforms.Normalize(NORM_MEAN, NORM_STD)(I)
        
        return I, mask


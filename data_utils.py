from pycocotools.coco import COCO 
import numpy as np 
import skimage.io as io 
#from torch.utils.data import Dataset


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
    
    def get_sample_by_category_id(self.cat_ids):
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

        ann_ids = self.coco.getAnnIds(imgIds=img['id'], catIds=self.coco.getCatIds(catIds=cat_ids), iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        labels = [ann['segmentation'] for ann in anns]


        return I, labels


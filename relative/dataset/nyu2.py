import cv2
import numpy as np
from pathlib import Path
from torchvision.transforms import Compose
from dataset.transform import Resize, NormalizeImage, PrepareForNet
from torch.utils.data import DataLoader, Dataset

class NYUD(Dataset):
    def __init__(self, filenames_file,size=(224,224)):
        with open(filenames_file, 'r') as f:
                self.filenames = f.readlines()
        self.transform = Compose([
            Resize(
                width=size[0],
                height=size[1],
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([]))

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        
        image_path = sample_path.split()[0]
        depth_path = sample_path.split()[1]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) 
        depth = np.asarray(depth, dtype=np.float32) / 1000.0 # in meters
        
        sample = dict(image=image, depth=depth)
        sample = self.transform(sample)
        sample['image_path'] = image_path
        return sample

    def __len__(self):
        return len(self.filenames)


def get_nyud_loader(data_dir_root, size=(224, 224)):
    dataset = NYUD(data_dir_root, size)
    return DataLoader(dataset, batch_size=1, shuffle=False,num_workers=4,pin_memory=True)


if __name__ == '__main__':
    dataset_root = Path(__file__).resolve().parents[1]
    valloader = get_nyud_loader(data_dir_root=str(dataset_root / 'dataset/splits/val_a800/nyu_val.txt'))
    for idx, data in enumerate(valloader):
        image,depth,valid_mask = data['image'],data['depth'],data['valid_mask']  # image torch.Size([1, 3, 224, 224]) depth torch.Size([1, 1, 480, 640])
        pass # debug point


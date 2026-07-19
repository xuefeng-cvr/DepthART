import cv2
import numpy as np
from torchvision.transforms import Compose
from dataset.transform import Resize, NormalizeImage, PrepareForNet
from torch.utils.data import DataLoader, Dataset

class DIODE(Dataset):
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
        sample_parts = self.filenames[idx].split()
        image_path, depth_path = sample_parts[:2]
        if len(sample_parts) >= 3:
            depth_mask_path = sample_parts[2]
        else:
            depth_mask_path = depth_path.replace("_depth.npy", "_depth_mask.npy")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        depth = np.load(depth_path)  # (768, 1024, 1) in meters
        depth = depth.transpose(2, 0, 1)  # 交换轴顺序 -> (1, 768, 1024)
        eval_mask = np.load(depth_mask_path)
        
        sample = dict(image=image, depth=depth)
        sample = self.transform(sample)
        sample['image_path'] = image_path
        sample['eval_mask'] = eval_mask
        return sample

    def __len__(self):
        return len(self.filenames)


def get_diode_loader(data_dir_root, size=(448, 448)):
    dataset = DIODE(data_dir_root, size)
    return DataLoader(dataset, batch_size=1, shuffle=False,num_workers=4,pin_memory=True)

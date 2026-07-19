import cv2
import numpy as np
from torchvision.transforms import Compose
from dataset.transform import Resize, NormalizeImage, PrepareForNet
from torch.utils.data import DataLoader, Dataset

class DDAD(Dataset):
    def __init__(self, filenames_file,size=(448,448)):
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
        depth = np.asarray(depth, dtype=np.float32) / 256.0 # in meters
        
        sample = dict(image=image, depth=depth)
        sample = self.transform(sample)
        sample['image_path'] = image_path
        return sample
    
    
    def __len__(self):
        return len(self.filenames)


def get_ddad_loader(data_dir_root, size=(448, 448)):
    dataset = DDAD(data_dir_root, size)
    return DataLoader(dataset, batch_size=1, shuffle=False,num_workers=4,pin_memory=True)



if __name__ == '__main__':
    from torch.utils.data import DataLoader
 
    filelist_path = "dataset/splits/val_800/ddad_val.txt"
    valloader = get_ddad_loader(filelist_path,(448, 448))
    for i, sample in enumerate(valloader):
        img, depth  = sample['image'].cuda(), sample['depth'].cuda() 
        pass
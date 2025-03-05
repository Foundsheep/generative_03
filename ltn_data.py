import lightning as L
import torch
from datasets import load_dataset
import numpy as np
from PIL import Image

from utils import get_transforms, convert_3_channel_to_1_channel
from args_default import Config

class CustomDS(torch.utils.data.Dataset):
    def __init__(self, dataset_repo, height, width, plate_dict_path, split="train"):
        super().__init__()
        self.ds = load_dataset(dataset_repo, split=split)
        self.transforms = get_transforms(height, width, plate_dict_path)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        image = self.ds[idx]["image"]
        plate_count = self.ds[idx]["plate_count"]
        rivet = self.ds[idx]["rivet"]
        die = self.ds[idx]["die"]
        upper_type = self.ds[idx]["upper_type"]
        upper_thickness = self.ds[idx]["upper_thickness"]
        middle_type = self.ds[idx]["middle_type"]
        middle_thickness = self.ds[idx]["middle_thickness"]
        lower_type = self.ds[idx]["lower_type"]
        lower_thickness = self.ds[idx]["lower_thickness"]
        head_height = self.ds[idx]["head_height"]

        # # image from 3 channel to 1 channel
        # image = convert_3_channel_to_1_channel(image)
        
        # transform
        image =  self.transforms["image"]["train"](image=np.array(image))[
            "image"
        ]
        plate_count = self.transforms["plate_count"](plate_count)
        rivet = self.transforms["rivet"](rivet)
        die = self.transforms["die"](die)
        upper_type = self.transforms["upper_type"](upper_type)
        upper_thickness = self.transforms["upper_thickness"](upper_thickness)
        middle_type = self.transforms["middle_type"](middle_type)
        middle_thickness = self.transforms["middle_thickness"](middle_thickness)
        lower_type = self.transforms["lower_type"](lower_type)
        lower_thickness = self.transforms["lower_thickness"](lower_thickness)
        head_height = self.transforms["head_height"](head_height)
        
        return (
            image, plate_count, rivet, die, upper_type, upper_thickness, 
            middle_type, middle_thickness, lower_type, lower_thickness, head_height
        )


class CustomDM(L.LightningDataModule):
    def __init__(self, dataset_repo, height, width, batch_size, shuffle, num_workers, plate_dict_path, is_full_data=True):
        super().__init__()
        self.dataset_repo = dataset_repo
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.is_full_data = is_full_data
        self.plate_dict_path = plate_dict_path
    
    def prepare_data(self):
        load_dataset(self.dataset_repo)
        
    def setup(self, stage):
        if self.is_full_data:
            self.ds_train = CustomDS(self.dataset_repo, self.height, self.width, self.plate_dict_path, "train[:30]+train[200:230]")
            
            # originally [90%:]. However, changed it to the front data to see how it works in a 2-plated combination in logs.
            self.ds_val = CustomDS(self.dataset_repo, self.height, self.width, self.plate_dict_path, "train[5:7]+train[225:227]")
        else:
            self.ds_train = CustomDS(self.dataset_repo, self.height, self.width, self.plate_dict_path, "train[:90%]")
            self.ds_val = CustomDS(self.dataset_repo, self.height, self.width, self.plate_dict_path, "train[90%:]")
            
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
    
def collate_fn(batch):
    return {
        "image": torch.stack([x[0] for x in batch]),
        "plate_count": torch.stack([x[1] for x in batch]),
        "rivet": torch.stack([x[2] for x in batch]),
        "die": torch.stack([x[3] for x in batch]),
        "upper_type": torch.stack([x[4] for x in batch]),
        "upper_thickness": torch.stack([x[5] for x in batch]),
        "middle_type": torch.stack([x[6] for x in batch]),
        "middle_thickness": torch.stack([x[7] for x in batch]),
        "lower_type": torch.stack([x[8] for x in batch]),
        "lower_thickness": torch.stack([x[9] for x in batch]),
        "head_height": torch.stack([x[10] for x in batch]),
    }

if __name__ == "__main__":
    dataset_repo = "DJMOON/hm_spr_01_04_640_480_default"
    height = 480
    width = 640
    plate_dict_path = "/mnt/c/Users/msi/Desktop/spr_datasets_01_04_resized_default/plate_dict.json"
    batch_size = 5
    shuffle = True
    num_workers = 2
    
    ds = CustomDS(dataset_repo, height, width, plate_dict_path)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    # dm = CustomDM(dataset_repo, height, width, batch_size, shuffle, num_workers)

    # print(dm)
    # print(dm.train_dataloader())
    print(len(dl))
    d = next(iter(dl))
    print(d)
    print(d["image"].size())
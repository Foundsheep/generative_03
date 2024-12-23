from diffusers import DDPMScheduler, DDIMScheduler, DDPMParallelScheduler, DDIMParallelScheduler, AmusedScheduler, DDPMWuerstchenScheduler, DDIMInverseScheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
# from torcheval.metrics import FrechetInceptionDistance
import json
import datetime
from PIL import Image
import numpy as np
from copy import deepcopy

def get_scheduler(scheduler_name):
    scheduler = None
    if scheduler_name == "DDPMScheduler":
        scheduler = DDPMScheduler()
    elif scheduler_name == "DDIMScheduler":
        scheduler = DDIMScheduler()
    elif scheduler_name == "DDPMParallelScheduler":
        scheduler = DDPMParallelScheduler()
    elif scheduler_name == "DDIMParallelScheduler":
        scheduler = DDIMParallelScheduler()
    elif scheduler_name == "AmusedScheduler":
        scheduler = AmusedScheduler()
    elif scheduler_name == "DDPMWuerstchenScheduler":
        scheduler = DDPMWuerstchenScheduler()
    elif scheduler_name == "DDIMInverseScheduler":
        scheduler = DDIMInverseScheduler()
    else:
        raise ValueError(f"scheduler name should be given, but [{scheduler_name = }]")
    return scheduler

def normalise_to_minus_one_and_one(x, x_min, x_max):
    normalised = (x - x_min) / x_max # to [0, 1]
    normalised = normalised * 2 - 1 # to [-1, 1]
    return normalised

def get_plate_dict(plate_dict_path):
    with open(plate_dict_path, "r") as f:
        plate_dict = json.load(f)
    return plate_dict

def get_transforms(height: int, width: int, plate_dict_path: str):
    plate_dict = get_plate_dict(plate_dict_path)
    return {
        "image": {
            "train": A.Compose(
                [                  
                    # interpolation=0 means cv2.INTER_NEAREST.
                    # default value is 1(cv2.INTER_LINEAR), which causes the array to have 
                    # other values from those already in the image
                    A.Resize(height=height, width=width, interpolation=0),
                    A.Normalize(mean=0.5, std=0.5), # make a range of [-1, 1]
                    ToTensorV2(),
                ]
            ),
            "val": A.Compose(
                [                  
                    # interpolation=0 means cv2.INTER_NEAREST.
                    # default value is 1(cv2.INTER_LINEAR), which causes the array to have 
                    # other values from those already in the image
                    A.Resize(height=height, width=width, interpolation=0),
                    A.Normalize(mean=0.5, std=0.5), # make a range of [-1, 1]
                    ToTensorV2(),
                ]
            ),
        },
        "plate_count": lambda x: torch.Tensor([normalise_to_minus_one_and_one(x, min(plate_dict["plate_count"]), max(plate_dict["plate_count"]))]),
        "rivet": lambda x: torch.Tensor([plate_dict["rivet"][x]]),
        "die": lambda x: torch.Tensor([plate_dict["die"][x]]),
        "upper_type": lambda x: torch.Tensor([plate_dict["upper_type"][x]]),
        "upper_thickness": lambda x: torch.Tensor([normalise_to_minus_one_and_one(x, min(plate_dict["upper_thickness"]), max(plate_dict["upper_thickness"]))]),
        "middle_type": lambda x: torch.Tensor([plate_dict["middle_type"][x]]),
        "middle_thickness": lambda x: torch.Tensor([normalise_to_minus_one_and_one(x, min(plate_dict["middle_thickness"]), max(plate_dict["middle_thickness"]))]),
        "lower_type": lambda x: torch.Tensor([plate_dict["lower_type"][x]]),
        "lower_thickness": lambda x: torch.Tensor([normalise_to_minus_one_and_one(x, min(plate_dict["lower_thickness"]), max(plate_dict["lower_thickness"]))]),
        "head_height": lambda x: torch.Tensor([normalise_to_minus_one_and_one(x, min(plate_dict["head_height"]), max(plate_dict["head_height"]))]),
    }
    
def get_class_nums(plate_dict_path):
    plate_dict = get_plate_dict(plate_dict_path)
    
    rivet_num = len(plate_dict["rivet"])
    die_num = len(plate_dict["die"])
    upper_type_num = len(plate_dict["upper_type"])
    middle_type_num = len(plate_dict["middle_type"])
    lower_type_num = len(plate_dict["lower_type"])
    return [rivet_num, die_num, upper_type_num, lower_type_num]

def get_fid(fake_images, real_images):
    # fid = FrechetInceptionDistance(feature_dim=2048, device="cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=2048)
    fid.inception.to("cuda" if torch.cuda.is_available() else "cpu")
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute()

def normalise_to_zero_and_one_from_minus_one(x: torch.Tensor, to_numpy=True) -> np.ndarray:
    out = (x / 2 + 0.5).clamp(0, 1)

    out = out.cpu().permute(0, 2, 3, 1).numpy() if to_numpy else out.cpu()
    return out

def normalise_to_zero_and_one_from_255(x: torch.Tensor) -> torch.Tensor:
    return (x / 255.0).clamp(0, 1)

def save_image(images: np.ndarray) -> None:
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for idx, img in enumerate(images):
        img = (img * 255).round().astype("uint8")
        img_to_save = Image.fromarray(img)
        
        folder_str = f"./{timestamp}_inference"
        folder = Path(folder_str)
        if not folder.exists():
            folder.mkdir()
            print(f"{folder} made..!")
        img_to_save.save(str(folder / f"{str(idx).zfill(2)}.png"))
        print(f"........{idx}th image saved!")
        
        
def resize_to_original_ratio(images: np.ndarray, to_h: int, to_w: int) -> np.ndarray:
    if images.ndim == 3:
        images = np.array([images])
    elif images.ndim != 4:
        raise ValueError(f"{images.ndim = }, should be either 3 or 4")

    resize_func = A.Resize(height=to_h, width=to_w, interpolation=0)
    result = []
    for img in images:
        out = resize_func(image=img)["image"]
        result.append(out)
    return np.array(result)


BACKGROUND = [0, 0, 0]
LOWER = [255, 96, 55]
MIDDLE = [221, 255, 51]
RIVET = [61, 245, 61]
UPPER = [61, 61, 245]
COLOUR_NAMES = {
    "BACKGROUND": BACKGROUND,
    "LOWER": LOWER,
    "MIDDLE": MIDDLE,
    "RIVET": RIVET,
    "UPPER": UPPER,
    }
RIVET_DIAMETER = 7.75

def colour_quantisation(arr_original):
    arr = deepcopy(arr_original)
    colours = [BACKGROUND, LOWER, MIDDLE, RIVET, UPPER]

    # print(f"...before quantisation: {len(np.unique(arr)) = }")
    for w in range(arr.shape[0]):
        for h in range(arr.shape[1]):
            max_diff = 255 * 3
            temp_diff = 0
            current_pixel = arr[w, h]
            quantised_pixel = [255, 255, 255]
            set_channel_idx = None
            # print(current_pixel)
            
            # 있는지 확인
            matching_flag = False
            for c in colours:
                if (current_pixel == c).all():
                    matching_flag = True
            if matching_flag:
                continue       
                    
            for colour_idx, c in enumerate(colours):
                # print(f"[{COLOUR_NAMES[colour_idx]}] : {c}")
                
                for channel_idx in range(arr.shape[2]):
                    temp_diff += np.abs(int(current_pixel[channel_idx]) - int(c[channel_idx]))                   
                
                if temp_diff == 0:
                    continue
                
                elif temp_diff < max_diff:
                    # print(f"It's smaller!, [{colour_idx}] colour, [{COLOUR_NAMES[colour_idx]}]")
                    max_diff = temp_diff
                    quantised_pixel = colours[colour_idx]
                    set_channel_idx = colour_idx
                
                temp_diff = 0
                # print(f"[{max_diff = }]")
            arr[w, h] = quantised_pixel
            # print(f"before: {current_pixel}, after: {quantised_pixel} -> [{COLOUR_NAMES[set_channel_idx]}]")

    # print(f"...after quantisation: {len(np.unique(arr)) = }")
    return arr

def denormalise_from_minus_one_to_255(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1) * 127.5).to(dtype=torch.uint8)
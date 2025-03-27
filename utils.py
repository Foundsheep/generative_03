from diffusers import DDPMScheduler, DDIMScheduler, DDPMParallelScheduler, DDIMParallelScheduler, AmusedScheduler, DDPMWuerstchenScheduler, DDIMInverseScheduler, PNDMScheduler
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
from args_default import Config

BACKGROUND = [0, 0, 0]
RIVET = [61, 245, 61]
UPPER = [61, 61, 245] 
MIDDLE = [221, 255, 51]
LOWER = [255, 96, 55]
COLOUR_ORDER = [BACKGROUND, RIVET, UPPER, MIDDLE, LOWER]
MIN_VAL = 0
MAX_VAL = len(COLOUR_ORDER) - 1

COLOUR_NAMES = {
    "BACKGROUND": BACKGROUND,
    "RIVET": RIVET,
    "MIDDLE": MIDDLE,
    "UPPER": UPPER,
    "LOWER": LOWER,
    }
RIVET_DIAMETER = 7.75


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
    elif scheduler_name == "PNDMScheduler":
        scheduler = PNDMScheduler()
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
                    # A.Normalize(mean=0.0, std=1.0), # make a range of [0, 1]
                    ToTensorV2(),
                ]
            ),
            "val": A.Compose(
                [                  
                    A.Resize(height=height, width=width, interpolation=0),
                    A.Normalize(mean=0.5, std=0.5), # make a range of [-1, 1]
                    # A.Normalize(mean=0.0, std=1.0), # make a range of [0, 1]
                    ToTensorV2(),
                ]
            ),
        },
        "plate_count": lambda x: torch.Tensor([normalise_to_minus_one_and_one(x, 2, 3)]),
        # "plate_count": lambda x: torch.Tensor([normalise_to_minus_one_and_one(x, min(plate_dict["plate_count"]), max(plate_dict["plate_count"]))]),
        # "plate_count": lambda x: torch.Tensor([x]).to(dtype=torch.float),
        "rivet": lambda x: torch.Tensor([plate_dict["rivet"][x]]).long(),
        "die": lambda x: torch.Tensor([plate_dict["die"][x]]).long(),
        "upper_type": lambda x: torch.Tensor([plate_dict["plate_name_list"][x]]).long(),
        "upper_thickness": lambda x: torch.Tensor([normalise_to_minus_one_and_one(x, min(plate_dict["plate_thickness_list"]), max(plate_dict["plate_thickness_list"]))]),
        # "upper_thickness": lambda x: torch.Tensor([x]).to(dtype=torch.float),
        "middle_type": (
            lambda x: torch.Tensor([plate_dict["plate_name_list"][x]]).long()
            if x is not None
            else torch.Tensor([len(plate_dict["plate_name_list"])]).long()
        ),
        "middle_thickness": (
            lambda x: torch.Tensor([normalise_to_minus_one_and_one(x, min(plate_dict["plate_thickness_list"]), max(plate_dict["plate_thickness_list"]))])
            # lambda x: torch.Tensor([x]).to(dtype=torch.float)
            if x is not None
            else torch.Tensor([Config.NONE_TENSOR_VALUE])
        ),
        "lower_type": lambda x: torch.Tensor([plate_dict["plate_name_list"][x]]).long(),
        "lower_thickness": lambda x: torch.Tensor([normalise_to_minus_one_and_one(x, min(plate_dict["plate_thickness_list"]), max(plate_dict["plate_thickness_list"]))]),
        # "lower_thickness": lambda x: torch.Tensor([x]).to(dtype=torch.float),
        
        # head_height is not normalised to [-1, 1], since it's already almost in that range
        # "head_height": lambda x: torch.Tensor([normalise_to_minus_one_and_one(x, min(plate_dict["head_height"]), max(plate_dict["head_height"]))]),
        "head_height": lambda x: torch.Tensor([x]).to(dtype=torch.float),
    }
    
def minmax_normalise(img):
    return np.divide(np.subtract(img, MIN_VAL), MAX_VAL)
    
def convert_3_channel_to_1_channel(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    canvas = np.zeros((img.shape[0], img.shape[1]))
    for idx, colour in enumerate(COLOUR_ORDER):
        if idx == 0:
            continue
        x, y = np.where(np.all(img == colour, axis=-1))
        canvas[x, y] = idx
    return np.expand_dims(canvas, axis=2).astype(np.float32)


def denormalise_to_class_indices(img:torch.Tensor) -> torch.Tensor:
    return (img.clamp(0, 1) * 4).round()

def convert_1_channel_to_3_channel(img: torch.Tensor) -> torch.Tensor:
    if isinstance(img, np.ndarray):
        img = torch.Tensor(img.transpose(2, 0, 1))
        
    if img.dim() == 3 and img.shape[0] == 1:
        img = img.squeeze()
    revert = torch.zeros((3, img.shape[0], img.shape[1])).to(device=img.device)
    for idx, colour in enumerate(COLOUR_ORDER):
        if idx == 0:
            continue
        x, y = torch.where(img == idx)
        revert[:, x, y] = torch.Tensor(colour).unsqueeze(dim=1).to(device=img.device)
    return revert

def convert_1_channel_to_3_channel_batch(batch, to_numpy=True):
    result = []
    for img in batch:
        img = convert_1_channel_to_3_channel(img)
        if to_numpy:
            img = img.permute(1, 2, 0).cpu().numpy()
        result.append(img)

    if to_numpy:
        result = np.stack(result, axis=0).astype(np.uint8)
    else:
        result = torch.stack(result, dim=0).to(dtype=torch.uint8)
    return result

def get_class_nums(plate_dict_path):
    plate_dict = get_plate_dict(plate_dict_path)
    
    rivet_num = len(plate_dict["rivet"])
    die_num = len(plate_dict["die"])
    
    # when plate names are shared across the plate types
    upper_type_num = len(plate_dict["plate_name_list"]) + 1
    middle_type_num = len(plate_dict["plate_name_list"]) + 1
    lower_type_num = len(plate_dict["plate_name_list"]) + 1

    # upper_type_num = len(plate_dict["upper_type"])
    # middle_type_num = len(plate_dict["middle_type"]) + 1
    # lower_type_num = len(plate_dict["lower_type"])
    return [rivet_num, die_num, upper_type_num, middle_type_num, lower_type_num]

def get_fid(fake_images:torch.Tensor, real_images:torch.Tensor, device):
    # fid = FrechetInceptionDistance(feature_dim=2048, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # if `normalize=True` it expects tensors ranging [0, 1]
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute()

def normalise_to_zero_and_one_from_minus_one(x: torch.Tensor, to_numpy=False) -> torch.Tensor:
    out = (x / 2 + 0.5).clamp(0, 1)

    out = out.cpu().permute(0, 2, 3, 1).numpy() if to_numpy else out
    return out

def denormalise_to_zero_and_one_from_255(x: torch.Tensor) -> torch.Tensor:
    return (x / 255.0).clamp(0, 1)

def save_image(images: np.ndarray) -> None:
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for idx, img in enumerate(images):
        img_to_save = Image.fromarray(img)
        
        folder_str = f"./{timestamp}_inference"
        folder = Path(folder_str)
        if not folder.exists():
            folder.mkdir()
            print(f"{folder} made..!")
        img_to_save.save(str(folder / f"{str(idx).zfill(2)}.png"))
        print(f"........{idx}th image saved!")
        
        
def resize_to_original_ratio(images: torch.Tensor, to_h: int, to_w: int) -> np.ndarray:
    if images.dim() == 3:
        images = np.array([images.permute(1, 2, 0)].cpu())
    elif images.dim() == 4:
        images = np.array(images.permute(0, 2, 3, 1).cpu())
    else:
        raise ValueError(f"{images.ndim = }, should be either 3 or 4")

    # 아직 digitisation되지 않은 상태이기 때문에 interpolation을 BILINEAR 같은 것으로 해도 무방
    resize_func = A.Resize(height=to_h, width=to_w)
    # resize_func = A.Resize(height=to_h, width=to_w, interpolation=0)
    return np.array([resize_func(image=img)["image"] for img in images])

def colour_quantisation_numpy(arr_original: np.ndarray) -> np.ndarray:
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


def colour_quantisation_torch(tensor_original: torch.Tensor) -> torch.Tensor:
    tensor = deepcopy(tensor_original)
    colours = [BACKGROUND, LOWER, MIDDLE, RIVET, UPPER]

    # print(f"...before quantisation: {len(torch.unique(tensor)) = }")
    for w in range(tensor.shape[1]):
        for h in range(tensor.shape[2]):
            max_diff = 255 * 3
            temp_diff = 0
            current_pixel = tensor[:, w, h].tolist()
            quantised_pixel = [255, 255, 255]
            set_channel_idx = None
            # print(current_pixel)

            matching_flag = any(current_pixel == c for c in colours)
            if matching_flag:
                continue       

            for colour_idx, c in enumerate(colours):
                # print(f"{COLOUR_NAMES[c] = }")

                for channel_idx in range(tensor.shape[0]):
                    temp_diff += abs(int(current_pixel[channel_idx]) - int(c[channel_idx]))                   

                if temp_diff == 0:
                    continue

                elif temp_diff < max_diff:
                    # print(f"It's smaller!, [{colour_idx}] colour, []")
                    max_diff = temp_diff
                    quantised_pixel = colours[colour_idx]
                    set_channel_idx = colour_idx

                temp_diff = 0
                # print(f"[{max_diff = }]")
            tensor[:, w, h] = torch.tensor(quantised_pixel)
                    # print(f"before: {current_pixel}, after: {quantised_pixel} -> [{COLOUR_NAMES[set_channel_idx]}]")

    # print(f"...after quantisation: {len(torch.unique(tensor)) = }")
    return tensor

def denormalise_from_minus_one_to_255(x: np.ndarray) -> np.ndarray:
    return ((x + 1) * 127.5).astype(np.uint8)

def denormalise_from_zero_one_to_255(x: np.ndarray) -> np.ndarray:
    return ((x * 255)).round().astype(dtype=np.uint8)
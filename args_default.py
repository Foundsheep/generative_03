from pathlib import Path


class Config:
    NONE_TENSOR_VALUR = -100
    PLATE_DICT_PATH = "./plate_dict.json"
    NUM_CONTINUOUS_CLASS_EMBEDS = 4
    TRAIN_NUM_STEPS = 1000
    UNET_SAMPLE_SIZE = [480, 640]
    UNET_BLOCK_OUT_CHANNELS = [32, 64, 128, 256]
    TRAIN_SCHEDULER_NAME = "DDPMScheduler"
    INFERENCE_SCHEDULER_NAME = "DDPMScheduler"
    INFERENCE_NUM_STEPS = 50
    INFERENCE_BATCH_SIZE = 2
    INFERENCE_HEIGHT = 480
    INFERENCE_WIDTH = 640
    LR = 0.001
    TRAIN_HEIGHT = 480
    TRAIN_WIDTH = 640
    TRAIN_BATCH_SIZE = 4
    TRAIN_SHUFFLE = True
    TRAIN_NUM_WORKERS = 8
    TRAIN_LOG_FOLDER = str(Path(__file__).absolute().parent)
    DATASET_REPO = "DJMOON/hm_spr_01_04_640_480_default"
    TRAIN_NUM_GPUS = 1
    MAX_EPOCHS = 2
    MIN_EPOCHS = 1
    LOG_EVERY_N_STEPS = 1
    SEED = 1105
    
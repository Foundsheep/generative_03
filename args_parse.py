import argparse
from args_default import Config

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--plate_dict_path", type=str, default=Config.PLATE_DICT_PATH)
    parser.add_argument("--num_continuous_class_embeds", type=int, default=Config.NUM_CONTINUOUS_CLASS_EMBEDS)
    parser.add_argument("--train_num_steps", type=int, default=Config.TRAIN_NUM_STEPS)
    parser.add_argument("--unet_sample_size", type=int, nargs=2, default=Config.UNET_SAMPLE_SIZE)
    parser.add_argument("--unet_block_out_channels", type=int, nargs="+", default=Config.UNET_BLOCK_OUT_CHANNELS)
    parser.add_argument("--train_scheduler_name", type=str, default=Config.TRAIN_SCHEDULER_NAME)
    parser.add_argument("--inference_scheduler_name", type=str, default=Config.INFERENCE_SCHEDULER_NAME)
    parser.add_argument("--inference_num_steps", type=int, default=Config.INFERENCE_NUM_STEPS)
    parser.add_argument("--inference_batch_size", type=int, default=Config.INFERENCE_BATCH_SIZE)
    parser.add_argument("--inference_height", type=int, default=Config.INFERENCE_HEIGHT)
    parser.add_argument("--inference_width", type=int, default=Config.INFERENCE_WIDTH)
    parser.add_argument("--lr", type=float, default=Config.LR)
    parser.add_argument("--dataset_repo", type=str, default=Config.DATASET_REPO)
    parser.add_argument("--train_height", type=int, default=Config.TRAIN_HEIGHT)
    parser.add_argument("--train_width", type=int, default=Config.TRAIN_WIDTH)
    parser.add_argument("--train_batch_size", type=int, default=Config.TRAIN_BATCH_SIZE)
    parser.add_argument("--train_shuffle", type=bool, default=Config.TRAIN_SHUFFLE)
    parser.add_argument("--train_num_workers", type=int, default=Config.TRAIN_NUM_WORKERS)
    parser.add_argument("--train_log_folder", type=str, default=Config.TRAIN_LOG_FOLDER)
    parser.add_argument("--train_num_gpus", type=int, default=Config.TRAIN_NUM_GPUS)
    parser.add_argument("--max_epochs", type=int, default=Config.MAX_EPOCHS)
    parser.add_argument("--min_epochs", type=int, default=Config.MIN_EPOCHS)
    parser.add_argument("--log_every_n_steps", type=int, default=Config.LOG_EVERY_N_STEPS)
    parser.add_argument("--fast_dev_run", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=Config.SEED)
    
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--predict", action=argparse.BooleanOptionalAction)
    parser.add_argument("--train", action=argparse.BooleanOptionalAction)

    return parser.parse_args()
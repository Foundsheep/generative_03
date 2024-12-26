import torch
import lightning as L
import datetime
from ltn_data import CustomDM
from ltn_model import CustomDDPM
import numpy as np
import random
import os
from utils import get_class_nums, get_transforms
from args_parse import get_args
from args_default import Config

torch.set_float32_matmul_precision("medium")

def train(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model = CustomDDPM(
        multi_class_nums=get_class_nums(args.plate_dict_path),
        num_continuous_class_embeds=args.num_continuous_class_embeds,
        train_num_steps=args.train_num_steps,
        train_batch_size=args.train_batch_size,
        unet_sample_size=args.unet_sample_size,
        unet_block_out_channels=args.unet_block_out_channels,
        train_scheduler_name=args.train_scheduler_name,
        inference_scheduler_name=args.inference_scheduler_name,
        inference_num_steps=args.inference_num_steps,
        inference_batch_size=args.inference_batch_size,
        inference_height=args.inference_height,
        inference_width=args.inference_width,
        lr = args.lr,
        is_train=True,
    )
    
    dm = CustomDM(
        dataset_repo=args.dataset_repo,
        height=args.train_height,
        width=args.train_width,
        batch_size=args.train_batch_size,
        shuffle=args.train_shuffle,
        num_workers=args.train_num_workers,
        plate_dict_path=args.plate_dict_path,
    )

    train_log_dir = f"{args.train_log_folder}/{timestamp}_batch{args.train_batch_size}_epochs{args.max_epochs}"

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.train_num_gpus,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=train_log_dir,
        fast_dev_run=args.fast_dev_run,
        strategy="ddp_find_unused_parameters_true",
        check_val_every_n_epoch=10
    )
    trainer.fit(model=model, datamodule=dm)
    print("*************** TRAINING DONE ***************")
    print("*********************************************")
    

def predict(args):
    
    # to resolve OOM error
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # os.environ["CUDA_HOME"] = "/usr/local/cuda"
    # os.environ["PATH"] = os.environ["CUDA_HOME"] + "/bin:" + os.environ["PATH"]
    # os.environ["LD_LIBRARY_PATH"] = os.environ["CUDA_HOME"] + "/lib64:" + os.environ["LD_LIBRARY_PATH"]
    # print(f"**** {os.environ['PYTORCH_CUDA_ALLOC_CONF'] = }")
        
    model = CustomDDPM.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        multi_class_nums=get_class_nums(args.plate_dict_path),
        num_continuous_class_embeds=args.num_continuous_class_embeds,
        train_num_steps=args.train_num_steps,
        train_batch_size=args.train_batch_size,
        unet_sample_size=args.unet_sample_size,
        unet_block_out_channels=args.unet_block_out_channels,
        train_scheduler_name=args.train_scheduler_name,
        inference_scheduler_name=args.inference_scheduler_name,
        inference_num_steps=args.inference_num_steps,
        inference_batch_size=args.inference_batch_size,
        inference_height=args.inference_height,
        inference_width=args.inference_width,
        lr = args.lr,
        is_train=False,
    )
    
    # conditions
    # TODO: inference_height and width are not used. To make it used,
    # modify the code to use them in resizing the output to a desired size
    transforms = get_transforms(
        height=args.inference_height,
        width=args.inference_width,
        plate_dict_path=args.plate_dict_path
    )
    plate_count = transforms["plate_count"](args.plate_count)
    rivet = transforms["rivet"](args.rivet)
    die = transforms["die"](args.die)
    upper_type = transforms["upper_type"](args.upper_type)
    upper_thickness = transforms["upper_thickness"](args.upper_thickness)
    middle_type = (
        transforms["middle_type"](args.middle_type)
        if args.middle_type is not None
        else torch.Tensor([Config.NONE_TENSOR_VALUE])
    )
    middle_thickness = (
        transforms["middle_thickness"](args.middle_thickness)
        if args.middle_thickness is not None
        else torch.Tensor([Config.NONE_TENSOR_VALUE])
    )
    lower_type = transforms["lower_type"](args.lower_type)
    lower_thickness = transforms["lower_thickness"](args.lower_thickness)
    head_height = transforms["head_height"](args.head_height)
    
    categorical_conds = (
        torch.stack([rivet, die, upper_type, lower_type])
        .to(device="cuda" if torch.cuda.is_available() else "cpu")
    )
    continuous_conds = (
        torch.stack([plate_count, upper_thickness, lower_thickness, head_height])
        .to(device="cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # OOM
    torch.cuda.empty_cache()
    model.eval()
    # model.to("cpu")
    torch.cuda.memory._record_memory_history()
    try:
        out = model(
            batch_size=args.inference_batch_size,
            categorical_conds=categorical_conds,
            continuous_conds=continuous_conds
        )
    except:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    finally:
        torch.cuda.memory._dump_snapshot("memory_snapshot_completed.pickle")
    print("*************** INFERENCE DONE ***************")
    print("**********************************************")
    
if __name__ == "__main__":
    args = get_args()
    print("**********************************************")
    print(dict(vars(args)))
    print("**********************************************")

    if args.train and not args.predict:
        train(args)
    elif not args.train and args.predict:
        predict(args)
    else:
        raise ValueError("either '--predict' or '--train' should be declared")
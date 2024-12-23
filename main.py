import torch
import lightning as L
import datetime
from ltn_data import CustomDM
from ltn_model import CustomDDPM
import numpy as np
import random
from utils import get_class_nums
from args_parse import get_args

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
    
    out = model()
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
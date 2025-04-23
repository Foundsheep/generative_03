import torch
import torchvision
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from diffusers import UNet2DModel
from utils import *
from tqdm import tqdm
import time

class CustomDDPM(L.LightningModule):
    def __init__(
        self,
        multi_class_nums,
        num_continuous_class_embeds,
        train_num_steps,
        unet_sample_size,
        unet_block_out_channels,
        train_scheduler_name,
        train_batch_size,
        inference_scheduler_name,
        inference_num_steps,
        inference_batch_size,
        inference_height,
        inference_width,
        lr,
        is_train,
    ):
        super().__init__()
        self.multi_class_nums = multi_class_nums
        self.num_continuous_class_embeds=num_continuous_class_embeds
        self.train_num_steps=train_num_steps
        self.train_batch_size=train_batch_size
        self.unet_sample_size=unet_sample_size
        self.unet_block_out_channels=unet_block_out_channels
        
        self.unet = UNet2DModel(
            in_channels=3,
            out_channels=3,
            sample_size=self.unet_sample_size,
            block_out_channels=self.unet_block_out_channels,
            norm_num_groups=self.unet_block_out_channels[0],
            num_continuous_class_embeds=self.num_continuous_class_embeds,
            multi_class_nums=self.multi_class_nums,
            freq_shift=0.0,
        )
        self.train_scheduler = get_scheduler(train_scheduler_name)
        self.inference_scheduler = get_scheduler(inference_scheduler_name)
        self.inference_num_steps = inference_num_steps
        self.inference_batch_size = inference_batch_size
        self.inference_height = inference_height
        self.inference_width = inference_width
        
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)
        # # changed for sparse gradient
        # # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # self.optimizer = torch.optim.SGD(self.unet.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 10, gamma=0.99)
        self.loss_fn = torch.nn.functional.mse_loss
        self.is_train = is_train
        
        # save hparams
        self.save_hyperparameters()
        
    def shared_step(self, batch, stage):
        image, categorical_conds, continuous_conds = self.unfold_batch(batch)
        noise = torch.randn_like(image, device=self.device)
        timestep = torch.randint(self.train_scheduler.config.num_train_timesteps, (image.size(0), ), device=self.device)
        noisy_image = self.train_scheduler.add_noise(image, noise, timestep)
        
        outputs = self.unet(
            sample=noisy_image,
            timestep=timestep,
            multi_class_labels=categorical_conds,
            continuous_class_labels=continuous_conds
        )
        residual = outputs.sample
        
        loss = self.loss_fn(residual, noise)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        return loss
        
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        real_images, categorical_conds, continuous_conds = self.unfold_batch(batch)
        fake_images = self(real_images.shape[0], categorical_conds, continuous_conds, do_post_process=False, do_save_fig=False)
        
        # loss is calculated based on the output range [-1, 1]
        loss = self.loss_fn(fake_images.to(dtype=torch.float32), real_images.to(dtype=torch.float32))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # fid is calculated based on the output range [0, 1]
        real_images = normalise_to_zero_and_one_from_minus_one(real_images)
        fake_images = normalise_to_zero_and_one_from_minus_one(fake_images)
        fid = get_fid(fake_images, real_images, self.device)
        self.log("val_fid", fid, prog_bar=True, on_epoch=True, sync_dist=True)

        # log image
        tb = self.logger.experiment
        grid_fake = torchvision.utils.make_grid(fake_images)
        grid_real = torchvision.utils.make_grid(real_images)
        tb.add_image(
            "val_samples",
            grid_fake,
            self.global_step,
        )
        tb.add_image(
            "val_reals",
            grid_real,
            self.global_step,
        )
        return loss
        
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler
        }
    
    def forward(self, batch_size, categorical_conds, continuous_conds, do_post_process=False, do_save_fig=True):
        self.inference_scheduler.set_timesteps(self.inference_num_steps)
        
        # if [-1, 1] -> torch.randn
        # if [0, 1] -> torch.rand
        images = torch.randn(
            (
                batch_size,
                3,
                self.unet_sample_size[0],
                self.unet_sample_size[1]
            ),
            device=self.device
        )
        
        for t in tqdm(self.inference_scheduler.timesteps):
            outs = self.unet(
                sample=images, 
                timestep=t, 
                multi_class_labels=categorical_conds, 
                continuous_class_labels=continuous_conds,
            )
            images = self.inference_scheduler.step(outs.sample, t, images).prev_sample

            # if OOM occurs... at least try...
            del outs
            torch.cuda.empty_cache()
        
        if do_post_process:
            images = resize_to_original_ratio(images, self.inference_height, self.inference_width)
            images = denormalise_from_minus_one_to_255(images)
            # images = [colour_quantisation_numpy(img) for out in images]

            if do_save_fig:
                self.save_generated_image(images)
        return images
    
    def configure_callbacks(self):
        checkpoint_save_last = ModelCheckpoint(
            save_last=True,
            filename="{epoch}-{step}-{train_loss:.4f}_last"
        )
        
        checkpoint_save_per_250 = ModelCheckpoint(
            save_top_k=-1,
            every_n_epochs=250,
            filename="{epoch}-{step}-{train_loss:.4f}_per_250"
        )
        
        return [checkpoint_save_last, checkpoint_save_per_250]
        
    def count_parameters(self):
        num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{num:,}")
        return num
    
    def unfold_batch(self, batch):
        image = batch["image"]
        plate_count = batch["plate_count"]
        rivet = batch["rivet"]
        die = batch["die"]
        upper_type = batch["upper_type"]
        upper_thickness = batch["upper_thickness"]
        middle_type = batch["middle_type"]
        middle_thickness = batch["middle_thickness"]
        lower_type = batch["lower_type"]
        lower_thickness = batch["lower_thickness"]
        head_height = batch["head_height"]
        
        # categorical_conds should have a shape of (N of conds, BS)
        # since it goes into the embedding layer in for loop. e.g.) for c in conds: em_layer[i](c)
        categorical_conds = torch.stack([rivet, die, upper_type, lower_type, middle_type], dim=0)
        
        # plate_count removed for its redundancy
        continuous_conds = torch.stack([upper_thickness, lower_thickness, middle_thickness, head_height], dim=1)
        # continuous_conds = torch.stack([plate_count, upper_thickness, lower_thickness, middle_thickness, head_height])        
        return image, categorical_conds, continuous_conds
    
    def save_generated_image(self, batch_outs):
        save_image(batch_outs)
    
if __name__ == "__main__":
    ddpm = CustomDDPM(
        multi_class_nums=[10, 5, 20, 3],
        num_continuous_class_embeds=5,
        num_train_steps=1,
        unet_sample_size=[480, 640],
        unet_block_out_channels=[32, 64, 128, 256],
        scheduler_name="DDPMScheduler",
        lr=0.001
    )
    
    print(ddpm.unet)
    print(ddpm.count_parameters())
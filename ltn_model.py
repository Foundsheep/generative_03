import torch
import lightning as L
from diffusers import UNet2DModel
from utils import *
from tqdm import tqdm

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
            sample_size=self.unet_sample_size,
            block_out_channels=self.unet_block_out_channels,
            num_continuous_class_embeds=self.num_continuous_class_embeds,
            multi_class_nums=self.multi_class_nums
        )
        self.train_scheduler = get_scheduler(train_scheduler_name)
        self.inference_scheduler = get_scheduler(inference_scheduler_name)
        self.inference_num_steps = inference_num_steps
        self.inference_batch_size = inference_batch_size
        self.inference_height = inference_height
        self.inference_width = inference_width
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.99)
        self.loss_fn = torch.nn.functional.mse_loss
        self.is_train = is_train
        
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
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        return loss
        
    
    def training_step(self, batch):
        return self.shared_step(batch, "train")
    
    def validation_step(self, batch):
        real_image, categorical_conds, continuous_conds = self.unfold_batch(batch)
        real_image = real_image.to(dtype=torch.uint8, device=self.device)
        fake_image = self(categorical_conds, continuous_conds, to_save_fig=False)
        
        fake_image = torch.stack([
            torch.from_numpy(
                colour_quantisation(
                    denormalise_from_minus_one_to_255(f_img)
                    .cpu()
                    .permute(1, 2, 0)
                    .numpy()
                )
            ).permute(2, 0, 1)
            for f_img in fake_image
        ]).to(dtype=torch.uint8, device=self.device)
        
        print(f"... {fake_image.dtype = } / {real_image.dtype = }")
        fid = get_fid(fake_image, real_image)
        loss = self.loss_fn(fake_image, real_image)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_fid", fid, prog_bar=True, on_epoch=True)
        
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler
        }
    
    def forward(self, categorical_conds, continuous_conds, to_save_fig=True):
        self.inference_scheduler.set_timesteps(self.inference_num_steps)
        
        image = torch.randn(
            (
                self.train_batch_size if self.is_train else self.inference_batch_size,
                3,
                self.unet_sample_size[0],
                self.unet_sample_size[1]
            ),
            device=self.device
        )

        for t in tqdm(self.inference_scheduler.timesteps):
            outs = self.unet(
                sample=image, 
                timestep=t, 
                multi_class_labels=categorical_conds, 
                continuous_class_labels=continuous_conds,
            )
            image = self.inference_scheduler.step(outs.sample, t, image).prev_sample
        
        if to_save_fig:
            self.save_generated_image(image)
        return image
        
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
        
        categorical_conds = torch.stack([rivet, die, upper_type, lower_type])
        continuous_conds = torch.stack([plate_count, upper_thickness, lower_thickness, head_height])
        
        return image, categorical_conds, continuous_conds
    
    def save_generated_image(self, batch_outs):
        outs = normalise_to_zero_and_one_from_minus_one(batch_outs)
        outs = resize_to_original_ratio(outs, self.inference_height, self.inference_width)
        save_image(outs)
    
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
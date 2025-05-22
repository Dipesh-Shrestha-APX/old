# # # Placeholder file
# # # pipeline/training_pipeline.py
# # import torch
# # import torch.nn as nn
# # from torch.utils.data import DataLoader
# # from models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator
# # from data.dataset import VITSDataset, get_dataloader
# # from utils.logging import Logger
# # import yaml
# # import os
# # from typing import Dict

# # class TrainingPipeline:
# #     def __init__(self, config_path: str, manifest_file: str = None):
# #         try:
# #             self.config = self.load_config(config_path)
# #             if manifest_file is not None:
# #                 self.config['manifest_file'] = manifest_file
# #             self.logger = Logger(self.config['log_dir'])
# #             self.models = self.initialize_models()
# #             self.optimizers = self.initialize_optimizers()
# #             self.train_dataset = VITSDataset(
# #                 self.config['data_dir'],
# #                 self.config['manifest_file'],
# #                 self.config['sample_rate'],
# #                 self.config['n_mels']
# #             )
# #             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
# #             self.val_dataset = VITSDataset(
# #                 self.config['data_dir'],
# #                 self.config['val_manifest_file'],
# #                 self.config['sample_rate'],
# #                 self.config['n_mels']
# #             )
# #             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
# #         except Exception as e:
# #             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
# #             raise

# #     def load_config(self, config_path: str) -> Dict:
# #         try:
# #             with open(config_path, 'r') as f:
# #                 return yaml.safe_load(f)
# #         except Exception as e:
# #             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
# #             raise

# #     def initialize_models(self) -> Dict[str, nn.Module]:
# #         return {
# #             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']).cuda(),
# #             'posterior_encoder': PosteriorEncoder().cuda(),
# #             'flow': Flow().cuda(),
# #             'duration_predictor': DurationPredictor().cuda(),
# #             'generator': HiFiGANGenerator().cuda(),
# #             'discriminator': Discriminator(self.config['periods']).cuda()
# #         }

# #     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
# #         return {
# #             'gen': torch.optim.Adam(
# #                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
# #                 lr=self.config['lr']
# #             ),
# #             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
# #         }

# #     def save_checkpoint(self, epoch: int, path: str):
# #         try:
# #             state = {
# #                 'epoch': epoch,
# #                 'models': {k: v.state_dict() for k, v in self.models.items()},
# #                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
# #             }
# #             torch.save(state, path)
# #         except Exception as e:
# #             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
# #             raise

# #     def validate(self):
# #         """Validate the model on the validation set."""
# #         try:
# #             total_loss = 0
# #             for phonemes, mels in self.val_loader:
# #                 phonemes, mels = phonemes.cuda(), mels.cuda()
# #                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
# #                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
# #                 z_flow, _ = self.models['flow'](z)
# #                 mel_pred = self.models['generator'](z_flow)
# #                 recon_loss = nn.MSELoss()(mel_pred, mels)
# #                 total_loss += recon_loss.item()
# #             avg_loss = total_loss / len(self.val_loader)
# #             self.logger.log({'val_loss': avg_loss})
# #             return avg_loss
# #         except Exception as e:
# #             self.logger.logger.error(f"Validation failed: {str(e)}")
# #             raise

# #     def run(self):
# #         try:
# #             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
# #             for epoch in range(self.config['epochs']):
# #                 for phonemes, mels in self.train_loader:
# #                     phonemes, mels = phonemes.cuda(), mels.cuda()

# #                     # Generator forward
# #                     z_mu, z_logvar = self.models['posterior_encoder'](mels)
# #                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
# #                     z_flow, log_det = self.models['flow'](z)
# #                     text_embed = self.models['text_encoder'](phonemes)
# #                     durations = self.models['duration_predictor'](text_embed)
# #                     mel_pred = self.models['generator'](z_flow)
# #                     audio = self.models['generator'](mel_pred)

# #                     # Losses
# #                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
# #                     recon_loss = nn.MSELoss()(mel_pred, mels)
# #                     adv_loss = 0
# #                     d_loss = 0
# #                     real_out = self.models['discriminator'](mels.unsqueeze(1))
# #                     fake_out = self.models['discriminator'](audio)
# #                     for r, f in zip(real_out, fake_out):
# #                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
# #                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))

# #                     # Optimize generator
# #                     self.optimizers['gen'].zero_grad()
# #                     total_gen_loss = recon_loss + kl_loss + adv_loss
# #                     total_gen_loss.backward()
# #                     self.optimizers['gen'].step()

# #                     # Optimize discriminator
# #                     self.optimizers['disc'].zero_grad()
# #                     d_loss.backward()
# #                     self.optimizers['disc'].step()

# #                     # Log metrics
# #                     self.logger.log({
# #                         'recon_loss': recon_loss.item(),
# #                         'kl_loss': kl_loss.item(),
# #                         'adv_loss': adv_loss.item(),
# #                         'd_loss': d_loss.item()
# #                     })

# #                 # Validate after each epoch
# #                 val_loss = self.validate()
# #                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")

# #                 # Save checkpoint
# #                 if (epoch + 1) % 10 == 0:
# #                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
# #         except Exception as e:
# #             self.logger.logger.error(f"Training failed: {str(e)}")
# #             raise

# #     def evaluate(self, test_manifest: str = "data/csv/test.csv"):
# #         """Evaluate the model on the test set."""
# #         try:
# #             test_dataset = VITSDataset(
# #                 self.config['data_dir'],
# #                 test_manifest,
# #                 self.config['sample_rate'],
# #                 self.config['n_mels']
# #             )
# #             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
# #             total_loss = 0
# #             for phonemes, mels in test_loader:
# #                 phonemes, mels = phonemes.cuda(), mels.cuda()
# #                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
# #                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
# #                 z_flow, _ = self.models['flow'](z)
# #                 mel_pred = self.models['generator'](z_flow)
# #                 recon_loss = nn.MSELoss()(mel_pred, mels)
# #                 total_loss += recon_loss.item()
# #             avg_loss = total_loss / len(test_loader)
# #             self.logger.log({'test_loss': avg_loss})
# #             return avg_loss
# #         except Exception as e:
# #             self.logger.logger.error(f"Evaluation failed: {str(e)}")
# #             raise

# # if __name__ == "__main__":
# #     pipeline = TrainingPipeline("configs/config.yaml")
# #     pipeline.run()
# ############################################################################33
# # import torch
# # import torch.nn as nn
# # from torch.utils.data import DataLoader
# # from models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator
# # from data.dataset import VITSDataset, get_dataloader
# # from utils.logging import Logger
# # import yaml
# # import os
# # from typing import Dict

# # class TrainingPipeline:
# #     def __init__(self, config_path: str, manifest_file: str = None):
# #         try:
# #             self.config = self.load_config(config_path)
# #             if manifest_file is not None:
# #                 self.config['manifest_file'] = manifest_file
# #             self.logger = Logger(self.config['log_dir'])
# #             self.models = self.initialize_models()
# #             self.optimizers = self.initialize_optimizers()
# #             # Initialize train dataset
# #             self.train_dataset = VITSDataset(
# #                 self.config['data_dir'],
# #                 self.config['manifest_file']
# #             )
# #             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
# #             # Initialize validation dataset
# #             self.val_dataset = VITSDataset(
# #                 self.config['data_dir'],
# #                 self.config['val_manifest_file']
# #             )
# #             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
# #         except Exception as e:
# #             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
# #             raise

# #     def load_config(self, config_path: str) -> Dict:
# #         try:
# #             with open(config_path, 'r') as f:
# #                 config = yaml.safe_load(f)
# #             # Validate required config fields
# #             required_fields = ['data_dir', 'manifest_file', 'val_manifest_file', 'batch_size', 'log_dir', 'checkpoint_dir', 'epochs', 'lr', 'n_vocab', 'embed_dim', 'periods']
# #             for field in required_fields:
# #                 if field not in config:
# #                     raise ValueError(f"Missing required config field: {field}")
# #             return config
# #         except Exception as e:
# #             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
# #             raise

# #     def initialize_models(self) -> Dict[str, nn.Module]:
# #         return {
# #             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']).cuda(),
# #             'posterior_encoder': PosteriorEncoder().cuda(),
# #             'flow': Flow().cuda(),
# #             'duration_predictor': DurationPredictor().cuda(),
# #             'generator': HiFiGANGenerator().cuda(),
# #             'discriminator': Discriminator(self.config['periods']).cuda()
# #         }

# #     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
# #         return {
# #             'gen': torch.optim.Adam(
# #                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
# #                 lr=self.config['lr']
# #             ),
# #             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
# #         }

# #     def save_checkpoint(self, epoch: int, path: str):
# #         try:
# #             state = {
# #                 'epoch': epoch,
# #                 'models': {k: v.state_dict() for k, v in self.models.items()},
# #                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
# #             }
# #             torch.save(state, path)
# #         except Exception as e:
# #             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
# #             raise

# #     def validate(self):
# #         """Validate the model on the validation set."""
# #         try:
# #             total_loss = 0
# #             for phonemes, mels in self.val_loader:
# #                 phonemes, mels = phonemes.cuda(), mels.cuda()
# #                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
# #                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
# #                 z_flow, _ = self.models['flow'](z)
# #                 mel_pred = self.models['generator'](z_flow)
# #                 recon_loss = nn.MSELoss()(mel_pred, mels)
# #                 total_loss += recon_loss.item()
# #             avg_loss = total_loss / len(self.val_loader)
# #             self.logger.log({'val_loss': avg_loss})
# #             return avg_loss
# #         except Exception as e:
# #             self.logger.logger.error(f"Validation failed: {str(e)}")
# #             raise

# #     def run(self):
# #         try:
# #             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
# #             for epoch in range(self.config['epochs']):
# #                 for phonemes, mels in self.train_loader:
# #                     phonemes, mels = phonemes.cuda(), mels.cuda()

# #                     # Generator forward
# #                     z_mu, z_logvar = self.models['posterior_encoder'](mels)
# #                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
# #                     z_flow, log_det = self.models['flow'](z)
# #                     text_embed = self.models['text_encoder'](phonemes)
# #                     durations = self.models['duration_predictor'](text_embed)
# #                     mel_pred = self.models['generator'](z_flow)
# #                     audio = self.models['generator'](mel_pred)

# #                     # Losses
# #                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
# #                     recon_loss = nn.MSELoss()(mel_pred, mels)
# #                     adv_loss = 0
# #                     d_loss = 0
# #                     real_out = self.models['discriminator'](mels.unsqueeze(1))
# #                     fake_out = self.models['discriminator'](audio)
# #                     for r, f in zip(real_out, fake_out):
# #                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
# #                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))

# #                     # Optimize generator
# #                     self.optimizers['gen'].zero_grad()
# #                     total_gen_loss = recon_loss + kl_loss + adv_loss
# #                     total_gen_loss.backward()
# #                     self.optimizers['gen'].step()

# #                     # Optimize discriminator
# #                     self.optimizers['disc'].zero_grad()
# #                     d_loss.backward()
# #                     self.optimizers['disc'].step()

# #                     # Log metrics
# #                     self.logger.log({
# #                         'recon_loss': recon_loss.item(),
# #                         'kl_loss': kl_loss.item(),
# #                         'adv_loss': adv_loss.item(),
# #                         'd_loss': d_loss.item()
# #                     })

# #                 # Validate after each epoch
# #                 val_loss = self.validate()
# #                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")

# #                 # Save checkpoint
# #                 if (epoch + 1) % 10 == 0:
# #                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
# #         except Exception as e:
# #             self.logger.logger.error(f"Training failed: {str(e)}")
# #             raise

# #     def evaluate(self, test_manifest: str = "data/csv/test.csv"):
# #         """Evaluate the model on the test set."""
# #         try:
# #             test_dataset = VITSDataset(
# #                 self.config['data_dir'],
# #                 test_manifest
# #             )
# #             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
# #             total_loss = 0
# #             for phonemes, mels in test_loader:
# #                 phonemes, mels = phonemes.cuda(), mels.cuda()
# #                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
# #                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
# #                 z_flow, _ = self.models['flow'](z)
# #                 mel_pred = self.models['generator'](z_flow)
# #                 recon_loss = nn.MSELoss()(mel_pred, mels)
# #                 total_loss += recon_loss.item()
# #             avg_loss = total_loss / len(test_loader)
# #             self.logger.log({'test_loss': avg_loss})
# #             return avg_loss
# #         except Exception as e:
# #             self.logger.logger.error(f"Evaluation failed: {str(e)}")
# #             raise

# # if __name__ == "__main__":
# #     pipeline = TrainingPipeline("configs/config.yaml")
# #     pipeline.run()
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator
# from vits_nepali.data.dataset import VITSDataset, get_dataloader
# from vits_nepali.utils.logging import Logger
# import yaml
# import os
# from typing import Dict

# class TrainingPipeline:
#     def __init__(self, config_path: str, manifest_file: str = None):
#         try:
#             self.config = self.load_config(config_path)
#             if manifest_file is not None:
#                 self.config['manifest_file'] = manifest_file
#             self.logger = Logger(self.config['log_dir'])
#             self.models = self.initialize_models()
#             self.optimizers = self.initialize_optimizers()
#             # Initialize train dataset
#             self.train_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 self.config['manifest_file']
#             )
#             self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
#             # Initialize validation dataset
#             self.val_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 self.config['val_manifest_file']
#             )
#             self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
#             raise

#     def load_config(self, config_path: str) -> Dict:
#         try:
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#             # Validate required config fields
#             required_fields = ['data_dir', 'manifest_file', 'val_manifest_file', 'batch_size', 'log_dir', 'checkpoint_dir', 'epochs', 'lr', 'n_vocab', 'embed_dim', 'periods']
#             for field in required_fields:
#                 if field not in config:
#                     raise ValueError(f"Missing required config field: {field}")
#             return config
#         except Exception as e:
#             self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
#             raise

#     def initialize_models(self) -> Dict[str, nn.Module]:
#         return {
#             'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']).cuda(),
#             'posterior_encoder': PosteriorEncoder().cuda(),
#             'flow': Flow().cuda(),
#             'duration_predictor': DurationPredictor().cuda(),
#             'generator': HiFiGANGenerator().cuda(),
#             'discriminator': Discriminator(self.config['periods']).cuda()
#         }

#     def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
#         return {
#             'gen': torch.optim.Adam(
#                 sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
#                 lr=self.config['lr']
#             ),
#             'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
#         }

#     def save_checkpoint(self, epoch: int, path: str):
#         try:
#             state = {
#                 'epoch': epoch,
#                 'models': {k: v.state_dict() for k, v in self.models.items()},
#                 'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
#             }
#             torch.save(state, path)
#         except Exception as e:
#             self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
#             raise

#     def validate(self):
#         """Validate the model on the validation set."""
#         try:
#             total_loss = 0
#             for phonemes, mels in self.val_loader:
#                 phonemes, mels = phonemes.cuda(), mels.cuda()
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['generator'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred, mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(self.val_loader)
#             self.logger.log({'val_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Validation failed: {str(e)}")
#             raise

#     def run(self):
#         try:
#             os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
#             for epoch in range(self.config['epochs']):
#                 for phonemes, mels in self.train_loader:
#                     phonemes, mels = phonemes.cuda(), mels.cuda()

#                     # Generator forward
#                     z_mu, z_logvar = self.models['posterior_encoder'](mels)
#                     z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
#                     z_flow, log_det = self.models['flow'](z)
#                     text_embed = self.models['text_encoder'](phonemes)
#                     durations = self.models['duration_predictor'](text_embed)
#                     mel_pred = self.models['generator'](z_flow)
#                     audio = self.models['generator'](mel_pred)

#                     # Losses
#                     kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
#                     recon_loss = nn.MSELoss()(mel_pred, mels)
#                     adv_loss = 0
#                     d_loss = 0
#                     real_out = self.models['discriminator'](mels.unsqueeze(1))
#                     fake_out = self.models['discriminator'](audio)
#                     for r, f in zip(real_out, fake_out):
#                         adv_loss += nn.MSELoss()(f, torch.ones_like(f))
#                         d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))

#                     # Optimize generator
#                     self.optimizers['gen'].zero_grad()
#                     total_gen_loss = recon_loss + kl_loss + adv_loss
#                     total_gen_loss.backward()
#                     self.optimizers['gen'].step()

#                     # Optimize discriminator
#                     self.optimizers['disc'].zero_grad()
#                     d_loss.backward()
#                     self.optimizers['disc'].step()

#                     # Log metrics
#                     self.logger.log({
#                         'recon_loss': recon_loss.item(),
#                         'kl_loss': kl_loss.item(),
#                         'adv_loss': adv_loss.item(),
#                         'd_loss': d_loss.item()
#                     })

#                 # Validate after each epoch
#                 val_loss = self.validate()
#                 self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")

#                 # Save checkpoint
#                 if (epoch + 1) % 10 == 0:
#                     self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
#         except Exception as e:
#             self.logger.logger.error(f"Training failed: {str(e)}")
#             raise

#     def evaluate(self, test_manifest: str = "data/csv/test_phonemes.csv"):
#         """Evaluate the model on the test set."""
#         try:
#             test_dataset = VITSDataset(
#                 self.config['data_dir'],
#                 test_manifest
#             )
#             test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
#             total_loss = 0
#             for phonemes, mels in test_loader:
#                 phonemes, mels = phonemes.cuda(), mels.cuda()
#                 z_mu, z_logvar = self.models['posterior_encoder'](mels)
#                 z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar)
#                 z_flow, _ = self.models['flow'](z)
#                 mel_pred = self.models['generator'](z_flow)
#                 recon_loss = nn.MSELoss()(mel_pred, mels)
#                 total_loss += recon_loss.item()
#             avg_loss = total_loss / len(test_loader)
#             self.logger.log({'test_loss': avg_loss})
#             return avg_loss
#         except Exception as e:
#             self.logger.logger.error(f"Evaluation failed: {str(e)}")
#             raise

# if __name__ == "__main__":
#     pipeline = TrainingPipeline("configs/config.yaml")
#     pipeline.run()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from vits_nepali.models import TextEncoder, PosteriorEncoder, Flow, DurationPredictor, HiFiGANGenerator, Discriminator
from vits_nepali.data.dataset import VITSDataset, get_dataloader
from vits_nepali.utils.logging import Logger
import yaml
import os
from typing import Dict

class TrainingPipeline:
    def __init__(self, config_path: str, manifest_file: str = None):
        try:
            self.config = self.load_config(config_path)
            if manifest_file is not None:
                self.config['manifest_file'] = manifest_file
            self.logger = Logger(self.config['log_dir'])
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.models = self.initialize_models()
            self.optimizers = self.initialize_optimizers()
            # Initialize train dataset
            self.train_dataset = VITSDataset(
                self.config['data_dir'],
                self.config['manifest_file']
            )
            self.train_loader = get_dataloader(self.train_dataset, self.config['batch_size'])
            # Initialize validation dataset
            self.val_dataset = VITSDataset(
                self.config['data_dir'],
                self.config['val_manifest_file']
            )
            self.val_loader = get_dataloader(self.val_dataset, self.config['batch_size'], shuffle=False)
        except Exception as e:
            self.logger.logger.error(f"Failed to initialize TrainingPipeline: {str(e)}")
            raise

    def load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Validate required config fields
            required_fields = ['data_dir', 'manifest_file', 'val_manifest_file', 'batch_size', 'log_dir', 'checkpoint_dir', 'epochs', 'lr', 'n_vocab', 'embed_dim', 'periods']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required config field: {field}")
            return config
        except Exception as e:
            self.logger.logger.error(f"Failed to load config from {config_path}: {str(e)}")
            raise

    def initialize_models(self) -> Dict[str, nn.Module]:
        models = {
            'text_encoder': TextEncoder(self.config['n_vocab'], self.config['embed_dim']),
            'posterior_encoder': PosteriorEncoder(),
            'flow': Flow(),
            'duration_predictor': DurationPredictor(),
            'generator': HiFiGANGenerator(),
            'discriminator': Discriminator(self.config['periods'])
        }
        return {k: v.to(self.device) for k, v in models.items()}

    def initialize_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            'gen': torch.optim.Adam(
                sum([list(m.parameters()) for m in self.models.values() if m != self.models['discriminator']], []),
                lr=self.config['lr']
            ),
            'disc': torch.optim.Adam(self.models['discriminator'].parameters(), lr=self.config['lr'])
        }

    def save_checkpoint(self, epoch: int, path: str):
        try:
            state = {
                'epoch': epoch,
                'models': {k: v.state_dict() for k, v in self.models.items()},
                'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()}
            }
            torch.save(state, path)
        except Exception as e:
            self.logger.logger.error(f"Failed to save checkpoint to {path}: {str(e)}")
            raise

    def validate(self):
        """Validate the model on the validation set."""
        try:
            total_loss = 0
            for phonemes, mels in self.val_loader:
                phonemes, mels = phonemes.to(self.device), mels.to(self.device)
                z_mu, z_logvar = self.models['posterior_encoder'](mels)
                z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
                z_flow, _ = self.models['flow'](z)
                mel_pred = self.models['generator'](z_flow)
                recon_loss = nn.MSELoss()(mel_pred, mels)
                total_loss += recon_loss.item()
            avg_loss = total_loss / len(self.val_loader)
            self.logger.log({'val_loss': avg_loss})
            return avg_loss
        except Exception as e:
            self.logger.logger.error(f"Validation failed: {str(e)}")
            raise

    def run(self):
        try:
            os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
            for epoch in range(self.config['epochs']):
                for phonemes, mels in self.train_loader:
                    phonemes, mels = phonemes.to(self.device), mels.to(self.device)

                    # Generator forward
                    z_mu, z_logvar = self.models['posterior_encoder'](mels)
                    z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
                    z_flow, log_det = self.models['flow'](z)
                    text_embed = self.models['text_encoder'](phonemes)
                    durations = self.models['duration_predictor'](text_embed)
                    mel_pred = self.models['generator'](z_flow)
                    audio = self.models['generator'](mel_pred)

                    # Losses
                    kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
                    recon_loss = nn.MSELoss()(mel_pred, mels)
                    adv_loss = 0
                    d_loss = 0
                    real_out = self.models['discriminator'](mels.unsqueeze(1))
                    fake_out = self.models['discriminator'](audio)
                    for r, f in zip(real_out, fake_out):
                        adv_loss += nn.MSELoss()(f, torch.ones_like(f))
                        d_loss += nn.MSELoss()(r, torch.ones_like(r)) + nn.MSELoss()(f, torch.zeros_like(f))

                    # Optimize generator
                    self.optimizers['gen'].zero_grad()
                    total_gen_loss = recon_loss + kl_loss + adv_loss
                    total_gen_loss.backward()
                    self.optimizers['gen'].step()

                    # Optimize discriminator
                    self.optimizers['disc'].zero_grad()
                    d_loss.backward()
                    self.optimizers['disc'].step()

                    # Log metrics
                    self.logger.log({
                        'recon_loss': recon_loss.item(),
                        'kl_loss': kl_loss.item(),
                        'adv_loss': adv_loss.item(),
                        'd_loss': d_loss.item()
                    })

                # Validate after each epoch
                val_loss = self.validate()
                self.logger.logger.info(f"Epoch {epoch+1}: Validation Loss = {val_loss}")

                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(epoch, f"{self.config['checkpoint_dir']}/epoch_{epoch+1}.pt")
        except Exception as e:
            self.logger.logger.error(f"Training failed: {str(e)}")
            raise

    def evaluate(self, test_manifest: str = "data/csv/test_phonemes.csv"):
        """Evaluate the model on the test set."""
        try:
            test_dataset = VITSDataset(
                self.config['data_dir'],
                test_manifest
            )
            test_loader = get_dataloader(test_dataset, self.config['batch_size'], shuffle=False)
            total_loss = 0
            for phonemes, mels in test_loader:
                phonemes, mels = phonemes.to(self.device), mels.to(self.device)
                z_mu, z_logvar = self.models['posterior_encoder'](mels)
                z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_logvar).to(self.device)
                z_flow, _ = self.models['flow'](z)
                mel_pred = self.models['generator'](z_flow)
                recon_loss = nn.MSELoss()(mel_pred, mels)
                total_loss += recon_loss.item()
            avg_loss = total_loss / len(test_loader)
            self.logger.log({'test_loss': avg_loss})
            return avg_loss
        except Exception as e:
            self.logger.logger.error(f"Evaluation failed: {str(e)}")
            raise

if __name__ == "__main__":
    pipeline = TrainingPipeline("configs/config.yaml")
    pipeline.run()
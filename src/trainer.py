"""
Model trainer.
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/08/14
"""
import os
from collections import defaultdict

import wandb
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
try:
    from apex import amp
except:
    print("Not found Nvidia apex. Please ensure it has been installed.")

from utils import (clustering_by_representation,
                   clustering_metric,
                   classification_metric,
                   classify_via_vote,
                   seed_everything,
                   print_network,
                   get_masked)
from vis_result import vis_hidden4training
from optimizer import get_optimizer, get_lr_scheduler


class Trainer:
    def __init__(self,
                 model_cls,
                 args,
                 train_dataset,
                 valid_dataset=None,
                 device='cpu') -> None:
        self.args = args
        self.device = device

        # unsupervised or classification.
        self.with_gt = self.args.train.with_gt
        self.model_cls = model_cls

        # config train logs.
        self.save_embeddings = self.args.train.save_embeddings
        self.history = defaultdict(list)
        if self.save_embeddings != -1:
            self.tiny_dataset = torch.load(os.path.join(
                f'./experiments/tiny-data/{self.args.dataset.name}_tiny.plk'))
            self.tiny_labels = self.tiny_dataset['y']
        with open(os.path.join(self.args.train.log_dir, 'hparams.yaml'), 'w') as f:
            f.write(self.args.dump())
        # Dataset.
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.num_workers = self.args.train.num_workers
        # Set reproducible
        self.seed = self.args.seed

        self.evaluate_intervals = self.args.train.evaluate

        # get mask.
        self.masked = False if self.args.train.missing_ratio == 0 else True
        if self.masked:
            self.masks = get_masked(self.args.train.batch_size,
                                    self.args.backbone.input_shapes,
                                    self.args.train.missing_ratio)
            torch.save(self.masks, os.path.join(
                self.args.train.log_dir, 'data_masks'))

        # for show reconstruction image.
        self.is_reconstruction = self.args.train.reconstruction
        if self.is_reconstruction:
            dl = DataLoader(valid_dataset, 16, shuffle=True)
            self.recon_samples = next(iter(dl))[0]
            self.recon_samples = [x.to(self.device) for x in self.recon_samples]
            # torch.save(self.recon_samples, os.path.join(
            #     self.args.train.log_dir, 'recon_samples'))

    def train(self):
        for idx, t in enumerate(range(self.args.train_time)):
            # Initialize.
            self.initialize_train(idx)
            # Training.
            for epoch in range(self.args.train.epochs):
                # train a epoch
                self.train_a_epoch(epoch)
                # Evaluate.
                if self.valid_loader and (epoch % self.evaluate_intervals == 0):
                    self.evaluate(epoch)
                # Save embeddings.
                if self.save_embeddings != -1 and epoch % self.save_embeddings == 0:
                    self.record_embedding(epoch)
                # Adjust learning rate.
                self.scheduler.step()
                self.writer.add_scalar('lr', self.scheduler.get_last_lr()[
                                       0], global_step=epoch)
            # finish a training process, record logs and save model.
            self.finish_a_train()
        # The end, save all log.
        if self.args.train.save_log:
            torch.save(self.history, f"{self.args.train.log_dir}/history.dict")

    def initialize_train(self, run_t):
        # initialize wandb.
        self.wandb_run = wandb.init(project=self.args.project_name,
                                    config=self.args,
                                    name=f"{self.args.experiment}-{self.args.dataset.name}-edi:{self.args.experiment_id}-run:{run_t}",
                                    reinit=True
                                    )
        wandb.config.update({'seed': self.seed}, allow_val_change=True)
        if self.save_embeddings != -1:
            self.embeddings = {}
            self.embeddings['labels'] = self.tiny_labels
        self.log_dir = os.path.join(
            self.args.train.log_dir, f'seed:{self.seed}')
        self.writer = SummaryWriter(log_dir=self.log_dir)
        seed_everything(self.seed)
        self.history['seeds'].append(self.seed)
        self.history[f"seed:{self.seed}"] = {
            "acc": [], "p": [], "fscore": [], "loss": [],
            "nmi": [], "ari": [],
        }
        model = self.model_cls(self.args,
                               device=self.device)
        self.model = model.to(self.device, non_blocking=True)
        self.optimizer = get_optimizer(
            model.parameters(), lr=self.args.train.lr, op_name=self.args.train.optim)
        if self.args.train.fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.args.train.opt_level)
            print(f'Enable mixed precision.')
        self.scheduler = get_lr_scheduler(
            self.args.train.lr_scheduler, self.optimizer)
        if self.args.verbose:
            print_network(self.model)
        self.train_loader = DataLoader(self.train_dataset,
                                       self.args.train.batch_size,
                                       num_workers=self.num_workers,
                                       shuffle=True,
                                       pin_memory=True)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       self.args.train.batch_size,
                                       num_workers=self.num_workers,
                                       pin_memory=True)
        # for knn evaluation.
        self.memory_loader = DataLoader(self.train_dataset,
                                        batch_size=self.args.train.batch_size,
                                        shuffle=False, 
                                        num_workers=self.num_workers, 
                                        pin_memory=True)
        self.best_loss = torch.inf
        # experimental setting. watch model by wandb
        wandb.watch(self.model, log='all', log_graph=True, log_freq=15)

    def train_a_epoch(self, epoch):
        self.model.train()
        losses = []
        loss_parts = defaultdict(list)
        if self.args.verbose:
            pbar = tqdm(self.train_loader, ncols=0, unit=" batch")
        for data in self.train_loader:
            Xs, y = data
            if self.masked:
                # masked data
                Xs = [x.masked_fill(m, 0) for m, x in zip(self.masks, Xs)]
            Xs = [Xs[idx].to(self.device) for idx in range(self.args.views)]
            self.optimizer.zero_grad()
            if self.with_gt:
                y = y.to(self.device)
                loss, loss_part = self.model.get_loss(Xs, y, epoch=epoch)
            else:
                loss, loss_part = self.model.get_loss(Xs, epoch=epoch)

            if len(loss_part) > 0:
                for lp in loss_part:
                    loss_parts[lp[0]].append(lp[1])

            if self.args.train.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            loss_part_dict = {k: np.round(np.mean(v), 4)
                              for k, v in loss_parts.items()}
            if self.args.verbose:
                pbar.update()
                pbar.set_postfix(
                    tag='TRAIN',
                    epoch=epoch,
                    loss=f"{np.mean(losses):.4f}",
                    lr=self.scheduler.get_last_lr()[0],
                    best_loss=f"{self.best_loss:.4f}",
                    **loss_part_dict
                )

        cur_loss = np.mean(losses)
        # Save model.
        if cur_loss < self.best_loss:
            self.model.eval()
            self.best_loss = cur_loss
            torch.save(self.model.state_dict(), os.path.join(
                self.log_dir, 'best_model.pth'))

        self.writer.add_scalar(
            'train-loss', np.mean(losses), global_step=epoch)
        wandb.log({'train-loss': np.mean(losses)}, step=epoch)
        self.history[f'seed:{self.seed}']['loss'].append(np.mean(losses))
        # add loss parts.
        for k, v in loss_parts.items():
            v_ = np.mean(v)
            self.writer.add_scalar(k, v_, global_step=epoch)
            wandb.log({k: v_}, step=epoch)

        if self.args.verbose:
            pbar.close()

    def evaluate(self, epoch):
        self.model.eval()
        predicts = []
        ground_truth = []
        with torch.no_grad():
            # Display reconstruction samples.
            if self.is_reconstruction and self.model.can_reconstruction:
                recon_Xs = self.model.enc_dec(self.recon_samples)
                orignal_view = torch.cat([x.detach().cpu() for x in self.recon_samples])
                orignal_grid = make_grid(orignal_view)
                recon_view = torch.cat([x.detach().cpu() for x in recon_Xs])
                recon_grid = make_grid(recon_view)
                wandb.log({'traget': wandb.Image(orignal_grid),
                           'generated': wandb.Image(recon_grid)},
                          step=epoch)
                # torch.save(recon_Xs, os.path.join(
                #     self.log_dir, f'recon_samples_{epoch}'))

            for data in tqdm(self.valid_loader, desc='Feature extracting'):
                Xs, y = data
                if self.masked:
                    # masked data
                    bs = Xs[0].shape[0]
                    Xs = [x.masked_fill(m[:bs], 0)
                          for m, x in zip(self.masks, Xs)]
                Xs = [Xs[idx].to(self.device)
                      for idx in range(self.args.views)]
                ground_truth.append(y)
                if self.model.can_predict:
                    Z = self.model.predict(Xs)
                else:
                    Z = self.model.commonZ(Xs)
                predicts.append(Z)

            ground_truth = torch.concat(ground_truth, dim=-1).numpy()
            if self.with_gt:
                if self.model.can_predict:
                    predicts = torch.concat(
                        predicts, dim=-1).squeeze().detach().cpu().numpy()
                    acc, p, fscore = classification_metric(
                        ground_truth, predicts)
                else:
                    predicts = torch.vstack(
                        predicts).squeeze().detach().cpu().numpy()
                    size = len(predicts) // 2
                    acc, p, fscore = classify_via_vote(
                        predicts[:size], ground_truth[:size], predicts[size:], ground_truth[size:])
                if self.args.verbose:
                    print(
                        f"[Valid] ACC: {acc}, (P)recision: {p}, fscore: {fscore}")
                # Record.
                self.writer.add_scalar('test-acc', acc, global_step=epoch)
                self.writer.add_scalar('test-p', p, global_step=epoch)
                self.writer.add_scalar('test-fscore', fscore, global_step=epoch)
                wandb.log({"test-acc": acc, "test-p": p, "test-fscore": fscore}, step=epoch)
                self.history[f'seed:{self.seed}']['acc'].append(acc)
                self.history[f'seed:{self.seed}']['p'].append(p)
                self.history[f'seed:{self.seed}']['fscore'].append(fscore)
            else:
                if self.model.can_predict:
                    predicts = torch.concat(
                        predicts, dim=-1).squeeze().detach().cpu().numpy()
                    acc, nmi, ari, _, p, fscore = clustering_metric(
                        ground_truth, predicts)
                else:
                    predicts = torch.vstack(
                        predicts).squeeze().detach().cpu().numpy()
                    acc, nmi, ari, _, p, fscore = clustering_by_representation(
                        predicts, ground_truth)
                if self.args.verbose:
                    print(f"[Valid] ACC: {acc}, NMI: {nmi}, ARI: {ari}, (P)recision: {p}, fscore: {fscore}")
                # Record.
                self.writer.add_scalar('test-acc', acc, global_step=epoch)
                self.writer.add_scalar('test-nmi', nmi, global_step=epoch)
                self.writer.add_scalar('test-ari', ari, global_step=epoch)
                self.history[f'seed:{self.seed}']['acc'].append(acc)
                self.history[f'seed:{self.seed}']['nmi'].append(nmi)
                self.history[f'seed:{self.seed}']['ari'].append(ari)

                self.writer.add_scalar('test-p', p, global_step=epoch)
                self.writer.add_scalar('test-fscore', fscore, global_step=epoch)
                self.history[f'seed:{self.seed}']['p'].append(p)
                self.history[f'seed:{self.seed}']['fscore'].append(fscore)
                wandb.log({"test-acc": acc, "test-nmi": nmi, "test-ari": ari,
                           "test-p": p, "test-fscore": fscore}, step=epoch)
            
            # KNN test.
            total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
            for Xs, y in tqdm(self.memory_loader, desc='Feature extracting'):
                if self.masked:
                    # masked data
                    bs = Xs[0].shape[0]
                    Xs = [x.masked_fill(m[:bs], 0)
                          for m, x in zip(self.masks, Xs)]
                Xs = [Xs[idx].to(self.device)
                      for idx in range(self.args.views)]
                feature = self.model.commonZ(Xs)
                feature_bank.append(feature)
                # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            feature_labels = self.memory_loader.dataset.targets.clone().detach().to(self.device)
            # feature_labels = torch.tensor(self.memory_loader.dataset.targets, device=feature_bank.device)
            cluster = len(feature_labels.unique())
            # loop test data to predict the label by weighted knn search
            test_bar = tqdm(self.valid_loader)
            for Xs, y in test_bar:
                data, target = [Xs[idx].to(self.device) for idx in range(self.args.views)], y.to(self.device)
                feature = self.model.commonZ(data)
                data_size = data[0].size(0)

                total_num += data_size
                # compute cos similarity between each feature vector and feature bank ---> [B, N]
                sim_matrix = torch.mm(feature, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=self.args.train.knn_num, dim=-1)
                # [B, K]
                sim_labels = torch.gather(feature_labels.expand(data_size, -1), dim=-1, index=sim_indices)
                sim_weight = (sim_weight / self.args.train.knn_temperature).exp()

                # counts for each class
                one_hot_label = torch.zeros(data_size * self.args.train.knn_num, cluster, device=sim_labels.device)
                # [B*K, C]
                one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
                # weighted score ---> [B, C]
                pred_scores = torch.sum(one_hot_label.view(data_size, -1, cluster) * sim_weight.unsqueeze(dim=-1), dim=1)

                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                        .format(epoch, self.args.train.epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))
            
            total_top1 = (total_top1 / total_num) * 100
            total_top5 = (total_top5 / total_num) * 100
            wandb.log({"knn_acc@1": total_top1, "knn_acc@5": total_top5}, step=epoch)

    def record_embedding(self, epoch):
        with torch.no_grad():
            self.model.eval()
            x1, x2 = self.tiny_dataset['x1'], self.tiny_dataset['x2']
            x1, x2 = x1.to(self.device), x2.to(self.device)
            hs, z = self.model.extract_all_hidden([x1, x2])
            hs = [h.detach().cpu() for h in hs]
            z = z.detach().cpu()
            if epoch % 5 == 0:
                # For reduce the storage pressure, decide to set a constant value 5 to save embedding.
                self.embeddings[f'hidden_hs_{epoch}'] = hs
                self.embeddings[f'hidden_z_{epoch}'] = z
            # add figure to tensorboard.
            self.writer.add_figure("latent-vis",
                                   vis_hidden4training(
                                       hs[0], hs[1], self.tiny_labels, z),
                                   global_step=epoch)

    def finish_a_train(self):
        self.seed = torch.randint(5000, (1, )).item()
        self.writer.close()
        # save lastest model.
        self.model.eval()
        torch.save(self.model.state_dict(), os.path.join(
            self.log_dir, 'final_mode.pth'))
        if self.save_embeddings != -1:
            torch.save(self.embeddings, os.path.join(
                self.log_dir, 'embeddings'))
        # To finish wandb's logging for that run.
        self.wandb_run.finish()

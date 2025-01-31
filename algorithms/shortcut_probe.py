import torch
import torch.nn as nn
import torch.nn.functional as F
from .algorithm import Algorithm
from data.sampler import GroupBalancedSampler
from data.embed_dataset import EmbedDataset
from tqdm import tqdm
from utils import AverageMeter, BestMetric, Timer, time_str, log
from .utils import init_optimizer, init_scheduler, extract_feature_info
from .register import register_algorithm
import numpy as np
import os
from models.classifier import Classifier
from torch.utils.data import DataLoader
from .utils import RunningVariance


class SpuriousVectors(nn.Module):
    def __init__(self, n_base, fea_dim, variance=None):
        super(SpuriousVectors, self).__init__()
        if variance is not None:
            self.spu_vecs = nn.Parameter(torch.normal(0, torch.sqrt(variance).unsqueeze(0).expand(n_base, fea_dim)))
        else:
            self.spu_vecs = nn.Parameter(torch.normal(0, 0.05, size=(n_base, fea_dim)))
        self.n_base = n_base
        self.fea_dim = fea_dim

    def detect(self, x):
        v = self.spu_vecs
        w = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(v, v.T)+1.0), v), x.T)
        weights = w.T
        return weights
    
    def forward(self, x):
        weights = self.detect(x)
        delta_vecs = torch.matmul(weights, self.spu_vecs)
        return delta_vecs
    



@register_algorithm("shortcutprobe")
class shortcutprobe(Algorithm):
    def __init__(self, config):
        super(shortcutprobe, self).__init__(config)
        self._init_model()
        self._init_training()


    def _init_model(self):
        self.n_classes = self.datasets["train"].n_classes
        self.device = f"cuda:{self.config.gpu}"
        self.model = Classifier(self.config.backbone, self.n_classes, self.config.pretrained)
        self.spu_vecs = SpuriousVectors(self.config.n_base, self.model.num_features)
        if self.config.check_point:
            log(f"loading the model checkpoint from {self.config.erm_model}")
            if len(self.config.erm_model) > 0:
                log(f"ignoring the ERM trained model from {self.config.erm_model}")
            saved_dict = self.load_check_point(self.config.check_point)
            self.model.load_state_dict(saved_dict["model_sd"])
            self.spu_vecs.load_state_dict(saved_dict["spu_sd"])
        elif len(self.config.check_point) == 0 and len(self.config.erm_model) > 0:
            log(f"loading the ERM trained model from {self.config.erm_model}")
            saved_dict = self.load_check_point(self.config.erm_model)
            self.model.load_state_dict(saved_dict["model_sd"])
        else:
            raise ValueError("Please provide an ERM-trained model")

        self.model.to(self.device)
        self.spu_vecs.to(self.device)

    def _init_training(self):
        
        if self.config.last_layer:
            self.optimizer_cls = init_optimizer(self.model.fc, self.config.optimizer_cls, self.config.optimizer_cls_kwargs)
            if self.config.scheduler_cls == "none":
                self.scheduler_cls =  None
            else:
                self.scheduler_cls = init_scheduler(self.optimizer_cls, self.config.scheduler_cls, self.config.scheduler_cls_kwargs)
            self.optimizer = None
            self.scheduler = None
        else:
            self.optimizer = init_optimizer(self.model, self.config.optimizer, self.config.optimizer_kwargs)
            self.scheduler = None
            self.optimizer_cls = None
            self.scheduler_cls = None
            

        self.optimizer_vec = init_optimizer(self.spu_vecs, self.config.optimizer_vec, self.config.optimizer_vec_kwargs)
        self.scheduler_vec = None

        self.criterion = torch.nn.CrossEntropyLoss()
        if self.config.split_val < 1.0:
            self.sel_metrics = [("val_subset2_acc", True), ("val_subset2_worst_cls_acc", True), ("val_subset2_worst_group_acc", True), ("val_subset2_avg_cls_diff", False), ("spu_acc", False)]
        else:
            self.sel_metrics = [("val_acc", True), ("val_worst_cls_acc", True), ("val_worst_group_acc", True), ("val_avg_cls_diff", False), ("spu_acc", False)]
        self.best_meters = {self._get_split(m):BestMetric(max_val) for m, max_val in self.sel_metrics}

    


    def shortcut_detection(self):
        timer = Timer()
        self.model.eval()
        best_metric = -float("inf")
        model_path = os.path.join(self.output_dir, "spu_vec.pt")
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        for epoch in range(1, self.config.shortcutprobe1_epochs+1):
            self.spu_vecs.train()
            epoch_meters = {k:AverageMeter() for k in ["loss", "loss_main", "norm_vec", "acc_mis", "acc_cor", "kldist0", "kldist1"]}
            if self.scheduler_vec:
                curr_lr = self.scheduler_vec.get_last_lr()[0]
            else:
                curr_lr = self.config.optimizer_vec_kwargs["lr"]

            for batch in self.dataloaders[self._get_split("shortcut_train")]:
                x, y, g, a, pred = batch
                y1, y2 = torch.split(y, y.size(0)//2)
                pred1, pred2 = torch.split(pred, pred.size(0)//2)

                x = x.to(self.device)
                pred = pred.to(self.device)
                y1 = y1.to(self.device)
                y2 = y2.to(self.device)
                pred1 = pred1.to(self.device)
                pred2 = pred2.to(self.device)

                if self.config.last_layer:
                    fea = x
                else:
                    with torch.no_grad():
                        _, fea = self.model(x, True)
                fea1, fea2 = torch.split(x, x.size(0)//2)


                delta_vecs = self.spu_vecs(fea)
                logits_spu = self.model.fc(delta_vecs)
                loss_main = criterion(logits_spu, pred).mean() # minmize the loss
                
                # loss_main = loss_spu1 

                weights1 = self.spu_vecs.detect(fea1)
                weights2 = self.spu_vecs.detect(fea2)
                weights1_0 = weights1[0::2]
                weights1_1 = weights1[1::2]
                weights2_0 = weights2[0::2]
                weights2_1 = weights2[1::2]
                kldist0 = (F.softmax(weights2_0, dim=-1) * (F.log_softmax(weights2_0, dim=-1) - F.log_softmax(weights1_0, dim=-1))).sum(dim=-1).mean()
                kldist1 = (F.softmax(weights2_1, dim=-1) * (F.log_softmax(weights2_1, dim=-1) - F.log_softmax(weights1_1, dim=-1))).sum(dim=-1).mean()
                
                vec_norm = torch.norm(delta_vecs,dim=-1).mean()
                loss = loss_main + self.config.sem_reg * vec_norm
                acc_cor = (torch.argmax(logits_spu[0:len(logits_spu)//2], dim=-1) == y1).float().mean()
                acc_mis = (torch.argmax(logits_spu[len(logits_spu)//2:], dim=-1) == pred2).float().mean()
                

                self.optimizer_vec.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.spu_vecs.parameters(), 5.0)
                self.optimizer_vec.step()

                epoch_meters["loss"].update(loss.item())
                epoch_meters["loss_main"].update(loss_main.item())
                epoch_meters["kldist0"].update(kldist0.item())
                epoch_meters["kldist1"].update(kldist1.item())
                epoch_meters["acc_cor"].update(acc_cor.item(), len(logits_spu)//2)
                epoch_meters["acc_mis"].update(acc_mis.item(), len(logits_spu)//2)
                epoch_meters["norm_vec"].update(vec_norm.item())

            if self.config.split_val < 1.0:
                loader = self.dataloaders[self._get_split("val_subset2")]
            else:
                loader = self.dataloaders[self._get_split("val")]
            self.spu_vecs.eval()
            acc_meter = AverageMeter()
            with torch.no_grad():
                for batch in tqdm(loader, leave=False):
                    x, y, g, p = batch
                    x, y = x.to(self.device), y.to(self.device)
                    if self.config.last_layer:
                        fea = x
                    else:
                        _, fea = self.model(x)
                    delta_vecs = self.spu_vecs(fea)
                    logits = self.model.fc(delta_vecs)
                    acc = (torch.argmax(logits, dim=-1) == y).float().mean()
                    acc_meter.update(acc.item(), len(y))
            metric = acc_meter.avg
            if metric > best_metric:
                best_metric = metric
                model_dict = {"model_sd":self.spu_vecs.state_dict(), "metric":best_metric}
                tag = "(best)"
                torch.save(model_dict, model_path)
            else:
                tag = ""
            elapsed_time = timer.t()
            est_all_time = elapsed_time/epoch * self.config.shortcutprobe1_epochs

            train_state = {k:epoch_meters[k].avg for k in epoch_meters}
            msg = ', '.join([f"{k}:{v:.6f}" for k,v in train_state.items()])
            log(f"[Step 1: Epoch {epoch}] {msg} spu_acc:{metric:.6f}, lr:{curr_lr:.6f} {tag} ({time_str(elapsed_time)}/{time_str(est_all_time)})")
        if self.config.shortcutprobe1_epochs > 0:
        # load the best model from Phase 1
            log(f"Loading the best model {model_path} from Step 1")
            result_dict = torch.load(model_path, map_location=self.device)
            self.spu_vecs.load_state_dict(result_dict["model_sd"])

    def get_group_labels(self):
        dataset = self.datasets[self._get_split(self.train_split)]
        group_labels = []
        sim_arr = []
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, pin_memory=True, shuffle=False, num_workers=self.config.num_workers)
        self.spu_vecs.eval()
        with torch.no_grad():
            for x, y, _, _ in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                sims = self.spu_vecs.detect(x)
                max_vals, max_indices = torch.max(sims,dim=1)
                groups = max_indices + y * self.config.n_base
                group_labels.append(groups)
                sim_arr.append(max_vals)
        group_labels = torch.cat(group_labels).detach().cpu().numpy()
        group_counts = (
            (
                torch.arange(self.config.n_base*self.n_classes).unsqueeze(1)
                == torch.from_numpy(group_labels)
            )
            .sum(1)
            .float()
        )
        total = sum(group_counts)
        msg = ''
        for g in range(len(group_counts)):
            y = g // self.config.n_base
            p = g % self.config.n_base
            gcount = group_counts[g]
            msg += f"Group{g} (y={y}, a={p}): {int(gcount)} ({gcount/total*100:.2f}%)\n"
        log(f"pseudo group labels:\n{msg}")
        sim_arr = torch.cat(sim_arr)
        indexes = torch.argsort(sim_arr)[-100:]
        reg_data = []
        reg_labels = []
        for i in indexes:
            x, y, _, _ = dataset[i]
            reg_data.append(x)
            reg_labels.append(y)
        reg_data = torch.stack(reg_data)
        reg_labels = torch.tensor(reg_labels)
        return group_labels, reg_data, reg_labels
    def bias_mitigation(self):
        timer = Timer()
        best_val = 0.0
        for epoch in range(1, self.config.shortcutprobe2_epochs+1):
            self.model.train()
            self.spu_vecs.eval()
            epoch_meters = {k:AverageMeter() for k in ["loss", "acc", "loss_spu", "loss_main"]}
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            if self.scheduler_cls:
                curr_lr = self.scheduler_cls.get_last_lr()[0]
            else:
                curr_lr = self.config.optimizer_cls_kwargs["lr"] if self.config.last_layer else self.config.optimizer_kwargs["lr"]
            if self.config.use_shortcutwalk_dataset:
                train_split = "train_joint"
            else:
                train_split = self.train_split
            for batch in self.dataloaders[self._get_split(train_split)]:
               
                if self.config.use_shortcutwalk_dataset:
                    x_mis, y_mis, g_mis, a_mis, y_mis_mis, x_cor, y_cor, g_cor, a_cor, y_cor_mis = batch
                    x_mis = x_mis.to(self.device)
                    x_cor = x_cor.to(self.device)
                    y_mis = y_mis.to(self.device)
                    y_cor = y_cor.to(self.device)
                    y_mis_mis = y_mis_mis.to(self.device)
                    y_cor_mis = y_cor_mis.to(self.device)
                    if self.config.last_layer:
                        fea_mis = x_mis
                        fea_cor = x_cor
                    else:
                        _, fea_mis = self.model(x_mis, True)
                        _, fea_cor = self.model(x_cor, True)
                    fea = torch.cat([fea_mis, fea_cor], dim=0)
                    logits_main = self.model.fc(fea)
                    loss_main = self.criterion(logits_main, torch.cat([y_mis, y_cor], dim=0))
                    delta_vecs = self.spu_vecs(fea)
                else:
                    x, y, g, a, pred = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                
                    if self.config.last_layer:
                        fea = x
                    else:
                        _, fea = self.model(x, True)
                    logits_main = self.model.fc(fea)
                    loss_main = criterion(logits_main, y).mean()
                    delta_vecs = self.spu_vecs(fea)

                logits_spu = self.model.fc(delta_vecs)
                # loss_spu = self.criterion(logits_spu, y)
                loss_spu = self.criterion(logits_spu, torch.cat([y_mis_mis, y_cor], dim=0))
                # loss_spu = (-F.softmax(logits_spu, dim=-1) * F.log_softmax(logits_spu, dim=-1)).sum(dim=-1).mean()
                if self.config.spu_reg == 0:
                    loss = loss_main
                else:
                    loss = self.config.spu_reg * loss_main / (loss_spu+1e-10)
                    # loss = loss_main + self.config.spu_reg * loss_core / (loss_spu+1e-10)
            


                if self.config.last_layer:
                    self.optimizer_cls.zero_grad()
                    loss.backward()
                    self.optimizer_cls.step()
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                epoch_meters["loss"].update(loss.item())
                
                # acc = (torch.argmax(logits_main, dim=-1) == y).float().mean()
                # epoch_meters["acc"].update(acc.item(), len(y))
                
                epoch_meters["loss_spu"].update(loss_spu.item())
                epoch_meters["loss_main"].update(loss_main.item())

            if self.config.last_layer and self.scheduler_cls:
                self.scheduler_cls.step()
            elif not self.config.last_layer and self.scheduler:
                self.scheduler.step()
          
            if epoch % self.config.eval_freq == 0:
                if self.config.split_val < 1.0:
                    result_dict = self.evaluate(self._get_split("val_subset2"))
                else:
                    result_dict = self.evaluate(self._get_split("val"))
                for metric, _ in self.sel_metrics:
                    metric = self._get_split(metric)
                    if self.best_meters[metric].add(result_dict[metric]):
                        self.save(epoch, self.best_meters[metric].get(), os.path.join(self.output_dir, f"best_{metric}_model.pt"))
                result_dict_test = self.evaluate(self._get_split("test"))
                result_dict.update(result_dict_test)
                train_state = {k:epoch_meters[k].avg for k in epoch_meters}
                msg = ', '.join([f"{k}:{v:.6f}" for k,v in train_state.items()])
                msg += ', '
                msg += ', '.join([f"{k}:{v:.6f}" for k,v in result_dict.items()])
                elapsed_time = timer.t()
                est_all_time = elapsed_time / epoch * self.config.shortcutprobe2_epochs
                log(f"[Step 2: Epoch {epoch}] {msg}, lr:{curr_lr:.6f} ({time_str(elapsed_time)}/{time_str(est_all_time)})")
    def evaluate(self, split):
        result_dict = super(shortcutprobe, self).evaluate(split)
        loader = self.dataloaders[split]
        self.model.eval()
        acc_meter = AverageMeter()
        with torch.no_grad():
            for batch in tqdm(loader, desc=split, leave=False):
                x, y, g, p = batch
                x, y = x.to(self.device), y.to(self.device)
                if "embed" in split:
                    fea = x
                else:
                    _, fea = self.model(x)
                delta_vecs = self.spu_vecs(fea)
                logits = self.model.fc(delta_vecs)
                acc = (torch.argmax(logits, dim=-1) == y).float().mean()
                acc_meter.update(acc.item(), len(y))
        result_dict.update({f"{self._get_split('spu_acc')}":acc_meter.avg})
        return result_dict
    
    def estimate_variance(self, train_split):
        running_var = RunningVariance()
        for batch in self.dataloaders[self._get_split(train_split)]:
            x, y, g, a = batch
            x = x.to(self.device)
            if self.config.last_layer:
                fea = x
            else:
                _, fea = self.model(x, True)
            for xi in x:
                running_var.update(xi)
        return running_var.variance()
            
    def train(self, output_dir, split="train"):
        timer = Timer()
        self.output_dir = output_dir
        self.train_split = split
        if self.config.last_layer:
            self.get_embed_loaders(self.train_split)
        for n_iter in range(self.config.n_processes):
            self.update_train_loaders(self.train_split)
            variance = self.estimate_variance(self.train_split)
            self.spu_vecs = SpuriousVectors(self.config.n_base, self.model.num_features, variance)
            self.spu_vecs.to(self.device)
            self.optimizer_vec = init_optimizer(self.spu_vecs, self.config.optimizer_vec, self.config.optimizer_vec_kwargs)
            self.shortcut_detection()
            self.bias_mitigation()
        self.save(self.config.shortcutprobe2_epochs, self.best_meters[self._get_split(self.sel_metrics[0][0])].get(), os.path.join(output_dir, "latest_model.pt"))
        elapsed_time = timer.t()
        log(f"total training time: {time_str(elapsed_time)}")

    def save(self, epoch, sel_metric, file_path):
        save_dict = {}
        save_dict["model_sd"] = self.model.state_dict()
        save_dict["sel_metric"] = sel_metric
        save_dict["config"] = self.config
        save_dict["optimizer"] = self.optimizer.state_dict() if self.optimizer else None
        save_dict["scheduler"] = self.scheduler.state_dict() if self.scheduler else None
        save_dict["epoch"] = epoch
        torch.save(save_dict, file_path)

    def test(self, output_dir, split=["test"], result_path=""):
        model_info = f"shortcutprobe {self.config.dataset} data_num:{self.total_data} {self.config.backbone} {self.config.train_split} train_ratio:{self.config.split_train:.2f} val_ratio:{self.config.split_val:.2f} n_base:{self.config.n_base} mis_ratio:{self.config.mis_ratios[0]:.2f} spu_reg:{self.config.spu_reg:.6f} len_reg:{self.config.sem_reg:.6f} seed:{self.config.seed}"
        if len(result_path) > 0:
            with open(result_path, "a") as fout:
                fout.write(model_info)
                fout.write('\n')
        for metric, _ in self.sel_metrics:
            metric = self._get_split(metric)
            result_dict = os.path.join(output_dir, f"best_{metric}_model.pt")
            saved_dict = self.load_check_point(result_dict)
            model_dict = saved_dict["model_sd"]
            sel_metric_val = saved_dict["sel_metric"]
            self.model.load_state_dict(model_dict)
            for sp in split:
                results = self.evaluate(self._get_split(sp))
                result_str = f"[{sp} ({metric}:{sel_metric_val:.6f})]: " + ', '.join([f"{k}:{results[k]:.6f}" for k in results])
                log(result_str) 
                if len(result_path) > 0:
                    with open(result_path, "a") as fout:
                        fout.write(result_str)
                        fout.write('\n')
      
            

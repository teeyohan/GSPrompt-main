#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, p, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        assert(set(tasks) == set(loss_ft.keys()))
        assert(set(tasks) == set(loss_weights.keys()))
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def loss_ft_mus(self, task_mus):
        # task-specific prompts
        sim_loss = 0.
        prompt_len = self.p.prompt_len
        for i in range(task_mus.shape[1] // prompt_len):
            for j in range(i+1, task_mus.shape[1] // prompt_len):
                vec_i = task_mus[:, i*prompt_len:(i+1)*prompt_len, :]
                vec_j = task_mus[:, j*prompt_len:(j+1)*prompt_len, :]
                sim_ij = F.cosine_similarity(vec_i, vec_j, dim=-1)
                sim_loss += sim_ij.mean()

        norm_loss = torch.norm(task_mus, p=2, dim=2).mean()
        total_loss = self.p.lambda_spe * sim_loss + self.p.lambda_spe_norm * norm_loss

        return total_loss

    def loss_ft_prompts(self, task_prompts):
        # task-generic prompts
        sim_loss = 0.
        prompt_len = self.p.prompt_len
        for i in range(task_prompts.shape[1] // prompt_len):
            for j in range(i+1, task_prompts.shape[1] // prompt_len):
                vec_i = task_prompts[:, i*prompt_len:(i+1)*prompt_len, :]
                vec_j = task_prompts[:, j*prompt_len:(j+1)*prompt_len, :]
                sim_ij = F.cosine_similarity(vec_i, vec_j, dim=-1)
                sim_loss -= sim_ij.mean()

        norm_loss = torch.norm(task_prompts, p=2, dim=2).mean()
        total_loss = self.p.lambda_sha * sim_loss + self.p.lambda_sha_norm * norm_loss

        return total_loss

    def forward(self, pred, gt, tasks):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in tasks}

        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in tasks]))
        out['total'] += self.loss_ft_mus(pred['task_mus'])
        out['total'] += self.loss_ft_prompts(pred['task_prompts'])

        return out

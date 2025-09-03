# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch.nn as nn
import torch.nn.functional as F

INTERPOLATE_MODE = 'bilinear'

class TaskPrompterWrapper(nn.Module):
    def __init__(self, p, backbone, heads):
        super(TaskPrompterWrapper, self).__init__()
        self.tasks = p.TASKS.NAMES

        self.backbone = backbone
        self.heads = heads 

        if 'dd_label_map_size' in p.keys():
            self.target_size = p.dd_label_map_size
        else:
            self.target_size = None

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}

        target_size = self.target_size if self.target_size is not None else img_size

        # TaskPrompter
        task_features, task_mus, task_prompts, info = self.backbone(x) 

        # Generate predictions
        out = {}
        for t in self.tasks:
            _task_fea = task_features[t]
            out[t] = F.interpolate(self.heads[t](_task_fea), target_size, mode=INTERPOLATE_MODE)

        out['task_mus'] = task_mus
        out['task_prompts'] = task_prompts
            
        return out

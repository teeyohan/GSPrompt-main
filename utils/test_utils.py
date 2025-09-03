# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from evaluation.evaluate_utils import PerformanceMeter
from tqdm import tqdm
from utils.utils import get_output, mkdir_if_missing
from evaluation.evaluate_utils import save_model_pred_for_one_task
import torch
import os

@torch.no_grad()
def test_phase(p, test_loader, model, criterion, epoch):
    all_tasks = [t for t in p.TASKS.NAMES]
    performance_meter = PerformanceMeter(p, all_tasks)

    model.eval()

    tasks_to_save = []
    for task in all_tasks:
        tasks_to_save.append(task)

    save_dirs = {task: os.path.join(p['save_dir'], 'iter' + str(epoch), task) for task in tasks_to_save}

    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)
    
    for i, batch in enumerate(tqdm(test_loader)):
        # Forward pass
        with torch.no_grad():
            images = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(non_blocking=True) for task in all_tasks}

            output = model(images) # to make ddp happy
        
            # Measure loss and performance
            performance_meter.update({t: get_output(output[t], t) for t in all_tasks}, 
                                    {t: targets[t] for t in all_tasks})

            for task in tasks_to_save:
                save_model_pred_for_one_task(p, i, batch, output, save_dirs, task, epoch=epoch)


    eval_results = performance_meter.get_score(verbose = True)

    return eval_results

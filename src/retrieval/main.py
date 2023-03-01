import os
from pathlib import Path
from tqdm import tqdm
from setproctitle import setproctitle
import pandas as pd

import torch

from open_clip import create_model_and_transforms
from retrieval.evaluation  import evaluate,get_metrics
from retrieval.params import parse_args
from retrieval.data import load_data

def print_metrics(metrics): 
    for metric in metrics : 
        if 'R@' in metric : 
            metrics[metric] = metrics[metric]*100
        metrics[metric] = round(metrics[metric],2)
    print(metrics)
        
def evaluate(dataloader,model,device,tops):
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, texts = batch
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            image_features, text_features, logit_scale = model(images, texts)
            return get_metrics(
                image_features=torch.cat([image_features]),
                text_features=torch.cat([text_features]),
                logit_scale=logit_scale,
                tops=tops
            )

def evaluate_per_shooting(args,model, preprocess,gt):
    def add_metrics(old, new):
        for metric in new:
            old[metric] =  old[metric]+new[metric]  if metric in old else 0 
        return old 
    metrics = {}    
    shoots = gt.groupby(args.csv_shooting_key)
    i =0
    for _, shoot in tqdm(shoots):
        dataloader = load_data(args,shoot,preprocess)
        metrics = add_metrics(metrics,evaluate(dataloader,model,args.device,args.tops))
        if i%args.log_rate == 0 and i!=0: 
            tmp_metric = {}
            for metric in metrics:
               tmp_metric[metric] = metrics[metric]/i
            print_metrics(tmp_metric)
        i+=1

    for metric in metrics:
        metrics[metric] = metrics[metric]/len(shoots)  
    
    return metrics


def evaluate_all(args,model, preprocess,gt):
    dataloader = load_data(args,gt,preprocess)
    return evaluate(dataloader,model,args.device,args.tops)
   
    
def main():
    setproctitle('[Evaluate CLIP]')

    args = parse_args()
    if not args : 
        print('Problem with arguments occured')
        return

    model, _, preprocess= create_model_and_transforms(
        args.network,
        pretrained= args.checkpoint if args.checkpoint is not None  else args.pretrained,
        device=args.device
    )
    
    gt = pd.read_csv(args.ground_truth_csv_path)

    metrics = evaluate_per_shooting(args,model, preprocess,gt) if args.per_shooting else evaluate_all(args,model, preprocess,gt)

    print_metrics(metrics)

if __name__ == "__main__":
    main()
    


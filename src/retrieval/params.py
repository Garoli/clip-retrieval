import argparse
import pandas as pd
import os 
import torch

def verify_columns_in_csv(dataframe, column):
    if not column in dataframe: 
        print(f'{column} not in csv !')
        return False
    return True
    
def parse_args():
    parser = argparse.ArgumentParser("evaluate clip retrieval")
    parser.add_argument('--ground_truth_csv_path', '-g', type=str, required=True,
                        help='Give an path to the csv with the ground truth')
    parser.add_argument('--csv_img_key','-ik', type=str, required=True,
                        help='Give the column name for the filenames')
    parser.add_argument('--csv_caption_key','-ck', type=str, required=True,
                        help='Give the column name for the labels')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='Give a path for the input image directory')
    parser.add_argument('--network', '-n', type=str, required=False,
                        help='Give network to use')
    parser.add_argument('--checkpoint','-c', type=str, required=False,
                        help='Give the checkpoint path')
    parser.add_argument('--per_shooting', '--ps', action='store_true', required=False)
    parser.add_argument('--csv_shooting_key','-sk', type=str, required=False,
                        help='Give the column name for the shooting ids')
    parser.add_argument('--csv_separator','-s', type=str,required=True,
                        help='Give the separator used in the csv')
    parser.add_argument('--workers','-w', type=int, default=4, required=False,
                        help='Give the number of workers')
    parser.add_argument('--device','-d', type=str, default='cpu', required=False,
                        help='Give the device')
    parser.add_argument('--pretrained','-p', type=str, default='openai', required=False,
                        help='Give the pretrained source')
    parser.add_argument('--log_rate','-lg', type=int, default=10, required=False,
                        help='Give the log rate')
    parser.add_argument('--tops','-t', nargs="*",type=int, default=[1,2,3,5,10], required=False,
                        help='Give the tops X to compute')
    
    args = parser.parse_args()

    if args.device : 
        if args.device == "cuda" and not torch.cuda.is_available(): 
            print("No cuda device available, will running on cpu")
        if args.device != "cuda" and args.device !="cpu":
            print("Please enter valid device")
            return


    args.network = args.network.replace('/', '-')
    if not os.path.isdir(args.input_dir):
        print(f'{args.input_dir} is not a dir')
        return 
    
    if not os.path.exists(args.ground_truth_csv_path):
        print(f'{args.ground_truth_csv_path} does not exist')
        return

    parser.add_argument('--val_data', type=str, const=args.ground_truth_csv_path, nargs='?',required=False)
    
    if args.checkpoint and not os.path.exists(args.checkpoint):
        print(f'{args.checkpoint} does not exist')
        return

    df = pd.read_csv(args.ground_truth_csv_path)
    if not verify_columns_in_csv(df, args.csv_img_key): return
    if not verify_columns_in_csv(df, args.csv_caption_key): return


    if not args.checkpoint and not args.network : 
        print('Choose zero shot model or give checkpoint to load')
        return
    
    if args.checkpoint and not os.path.exists(args.checkpoint):
        print(f'{args.checkpoint} does not exist')
        return

    if args.per_shooting and not args.csv_shooting_key: 
        verify_columns_in_csv(df,args.csv_shooting_key )
        print(f'per_shooting was {args.per_shooting} but no shooting_column_name was provided')
        return

    return args

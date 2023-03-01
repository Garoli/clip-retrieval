from open_clip import tokenize
from PIL import Image
from torch.utils.data import Dataset,DataLoader

class CsvDataset(Dataset):
    def __init__(self,df, transforms, img_key, caption_key, sep="\t"):
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
  
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = tokenize([str(f"A photo of a { self.captions[idx]}, a type of meal")])[0]
        return images, texts

def load_data(args, dataframe, preprocess):
    dataset = CsvDataset(
            dataframe,
            preprocess,  
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator)
    num_samples = len(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=num_samples,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return dataloader


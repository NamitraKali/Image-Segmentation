import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np

class ImgDataset():
    """
    Creates a PyTorch readable dataset for images

    Arguments:
        path (String, Path-string): contains the path to the train/validation/test directories in
            an image dataset
        bs (Int): the batch size for the model
        aug (List): a list of torchvision transforms that will augment the training set
        test_aug (List): a list of torchvision transforms that will augment the validation/test sets
    """
    def __init__(self, path, bs=1, aug=None, test_aug=None):
        self.path = path
        self.bs = bs
        self.aug = aug
        self.test_aug = test_aug
        
        self.train_dataset = datasets.ImageFolder(f'{path}/train', transform=aug)
        self.train = data.DataLoader(self.train_dataset, batch_size=bs, num_workers=2)
        
        try:
            self.test_dataset = datasets.ImageFolder(f'{path}/test',
                                                     transform=test_aug)
            self.test = data.DataLoader(self.test_dataset,
                                        batch_size=bs, num_workers=9)
        except:
            pass
        
        try:
            self.val_dataset = datasets.ImageFolder(f'{path}/val', transform=test_aug)
            self.val = data.DataLoader(self.val_dataset, batch_size=bs, num_workers=9)
        except:
            pass
        
    def apply_sampler(self, sampler):
        self.train_dataset = datasets.ImageFolder(f'{self.path}/train',
                                                  transform=self.aug)

        self.train = data.DataLoader(self.train_dataset, batch_size=self.bs,
                                     sampler=sampler, num_workers=2)


class TabularData(data.Dataset):
    """
    Creates a PyTorch readable dataset from a pandas dataframe

    Arguments:
        df (Pandas Dataframe): a Pandas dataframe that represents our dataset
        cat (String, List, optional): a list/string of the categorical columns in df
        y (String, List): a list/string of the dependant column(s)

    Output: A collection of PyTorch arrays
    """
    def __init__(self, df, cat, y):
        # Isolate continuous categorical columns
        self.df_conts = df.drop(cat, axis=1)
        self.df_conts = self.df_conts.drop(y, axis=1)

        # Isolate categorical columns
        self.df_cats = df[cat]

        # Isolate y columns
        self.df_y = df[y]

        self.conts = np.stack([c.values for n, c in self.df_conts.items()], axis=1)
        self.cats = np.stack([c.values for n, c in self.df_cats.items()], axis=1)
        self.y = np.stack([c.values for n, c in self.df_y], axis=1)
    
    def __len__(self): return len(self.y)
    
    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx], self.y[idx]]
    

def LoadTablularData(df, cat, y, bs=1, shuffle=False, workers=1):
    """
    Takes a pandas dataset and returns a DataLoader that can be fed into
    PyTorch model
    
    Arguments:
        df (Pandas Dataframe): a Pandas dataframe that represents our dataset
        cat (String, Listof(String), optional): a string/list of strings of the categorical columns in df
        y (String, Listof(String)): a string/list of strings of the dependant columns
        bs (int): the batch size for our model
        shuffle (Boolean): Whether or not we want to shuffle our dataset during training
        workers (Int): How many threads to process the data
    """
    dataset = TabularData(df, cat, y)
    return data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=workers)
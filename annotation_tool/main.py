from dataloader import DataLoader
from interface import Interface

if __name__ == '__main__':
    dataloader = DataLoader('datasets/train.json', output_dir='annotations')
    interface = Interface(dataloader)
from dataloader import DataLoader
from interface import Interface

if __name__ == '__main__':
    dataloader = DataLoader('../dataset_creation/batches/batch_0.json', output_dir='annotations')
    interface = Interface(dataloader)
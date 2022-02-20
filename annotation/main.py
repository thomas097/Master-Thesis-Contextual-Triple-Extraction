from dataloader import DatasetIO
from interface import Interface

if __name__ == '__main__':
    dataloader = DatasetIO('datasets/dailydialog.txt')
    interface = Interface(dataloader)

from dataloader import DatasetIO
from dependency_parser import DependencyTripleParser
from interface import Interface

if __name__ == '__main__':
    dataloader = DatasetIO('datasets/dailydialog.txt')
    parser = DependencyTripleParser()
    interface = Interface(dataloader, parser)

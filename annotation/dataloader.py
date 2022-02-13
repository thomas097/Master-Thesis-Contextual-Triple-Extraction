import os
from pathlib import Path
from nltk import word_tokenize
import json


class DatasetIO:
    def __init__(self, path, output_dir='annotations', sep='<eos>'):
        # Read dataset from file
        with open(path, 'r', encoding='utf-8') as file:
            self._lines = file.readlines()

        self._output_dir = output_dir
        self._dataset_name = Path(path).stem
        self._ptr = -1
        self._sep = sep

        # Set up directory to store annotations into
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    @property
    def _outfile(self):
        return '{}/{}_{}.json'.format(self._output_dir, self._dataset_name, str.zfill(str(self._ptr), 6))

    def next(self):
        # Search for line in dataset yet to be annotated
        self._ptr += 1
        while os.path.exists(self._outfile):
            self._ptr += 1

        # Check if done
        if self._ptr >= len(self._lines):
            return None

        # Return sample as list of tokenized turns
        dialog = self._lines[self._ptr]
        return [word_tokenize(turn.strip()) for turn in dialog.split(self._sep)]

    def save(self, annotation):
        with open(self._outfile, 'w') as file:
            json.dump(annotation, file)


if __name__ == '__main__':
    dataset = DatasetIO('dataset.txt')
    print(dataset.next())

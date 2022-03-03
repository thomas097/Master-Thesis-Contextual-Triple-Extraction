import os
import spacy
import json


class List:
    def __init__(self, items):
        self._items = items
        self._i = -1

    def next(self):
        self._i += 1
        if self._i >= len(self._items):
            self._i = len(self._items) - 1
        return self._items[self._i]

    def prev(self):
        self._i -= 1
        if self._i < 0:
            self._i = 0
        return self._items[self._i]


class DataLoader:
    def __init__(self, path, output_dir='annotations', sep='<eos>'):
        # Tokenizer
        self.__nlp = spacy.load("en_core_web_sm")

        # Read dataset from file
        with open(path, 'r', encoding='utf-8') as file:
            self._samples = List(json.load(file))

        self._history = []
        self._sample = None
        self._sep = sep

        # Set up directory to store annotations into
        self._output_dir = output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    @property
    def current_id(self):
        return self._sample['id']

    def _previously_annotated(self, sample):
        """ Checks whether sample was annotated in previous session
            (this sample can be skipped).
        """
        filename = self._output_dir + '/annotated_' + sample['id'] + '.json'
        return os.path.exists(filename)

    def _in_current_session(self, sample):
        """ Checks whether sample was annotated in current session.
        """
        return sample['id'] in self._history

    def next(self):
        """ Returns the next to-be annotated sample.
        """
        # Continue until unannotated sample is found
        self._sample = self._samples.next()
        while self._previously_annotated(self._sample):
            self._sample = self._samples.next()

        # Add to history
        self._history.append(self._sample['id'])

        # Return sample as list of tokenized turns
        return [[w.lower_ for w in self.__nlp(t.strip())] for t in self._sample['triplet'].split(self._sep)]

    def prev(self):
        """ Returns the previously annotated sample.
        """
        self._sample = self._samples.prev()
        while not self._in_current_session(self._sample):
            self._sample = self._samples.prev()

        # Return sample as list of tokenized turns
        return [[w.lower_ for w in self.__nlp(t.strip())] for t in self._sample['triplet'].split(self._sep)]

    def save(self, annotation):
        outfile = self._output_dir + '/annotated_' + self._sample['id'] + '.json'
        with open(outfile, 'w') as file:
            json.dump(annotation, file)


if __name__ == '__main__':
    dataset = DataLoader('datasets/train.json')
    print('next():', dataset.next())
    print('next():', dataset.next())
    print('next():', dataset.next())
    print('prev():', dataset.prev())
    print('next():', dataset.next())

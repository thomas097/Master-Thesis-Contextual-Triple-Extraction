import os
import spacy
import json
import tkinter as tk
from tkinter.filedialog import askopenfilename


class List:
    def __init__(self, items):
        self._items = items
        self._i = -1

    def __len__(self):
        return len(self._items)

    @property
    def index(self):
        return self._i

    def has_next(self):
        return self._i < len(self._items) - 1

    def next(self):
        """ Returns the next item in the list
        """
        self._i += 1
        if self._i >= len(self._items):
            self._i = len(self._items) - 1
        return self._items[self._i]

    def has_prev(self):
        return self._i > 0

    def prev(self):
        """ Goes back to previous item visited in the list
        """
        self._i -= 1
        if self._i < 0:
            self._i = 0
        return self._items[self._i]


class DataLoader:
    def __init__(self, path, output_dir='annotations', sep='<eos>'):
        # Tokenizer
        self.__nlp = spacy.load("en_core_web_sm")

        # Read 'sep'-separated dataset from file
        with open(path, 'r', encoding='utf-8') as file:
            self._dataset = List(json.load(file))

        self._history = []    # IDs of previously annotated samples
        self._current = None  # Sample currently being annotated
        self._sep = sep

        # Set up directory to store annotations into
        self._output_dir = output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    @property
    def progress(self):
        return '{}/{}'.format(self._dataset.index, len(self._dataset))

    @property
    def current_id(self):
        """ Returns the ID of the sample being annotated
        """
        return self._current['id']

    def _previously_annotated(self, sample):
        """ Checks whether sample was annotated in previous session
            (this sample can be skipped)
        """
        filename = self._output_dir + '/annotated_' + sample['id'] + '.json'
        return os.path.exists(filename)

    def _in_current_session(self, sample):
        """ Checks whether sample was annotated in current session
        """
        return sample['id'] in self._history

    def _tokenize(self, current):
        turns = current['triplet'].split(self._sep)
        return [[w.lower_ for w in self.__nlp(turn.strip())] for turn in turns]

    def next(self):
        """ Returns the next sample to be annotated
        """
        # Continue until unannotated sample is found
        self._current = self._dataset.next()
        while self._previously_annotated(self._current) and self._dataset.has_next():
            self._current = self._dataset.next()

        # Add new sample to history
        self._history.append(self._current['id'])

        # Return sample as list of tokenized turns
        return self._tokenize(self._current)

    def prev(self):
        """ Returns the previously annotated sample (within current session)
        """
        # Step down list until previous sample is found
        self._current = self._dataset.prev()
        while not self._in_current_session(self._current) and self._dataset.has_prev():
            self._current = self._dataset.prev()

        # Return sample as list of tokenized turns
        return self._tokenize(self._current)

    def save(self, annotation):
        outfile = self._output_dir + '/annotated_' + self._current['id'] + '.json'
        with open(outfile, 'w') as file:
            json.dump(annotation, file)


def filebrowser():
    root = tk.Tk()
    root.withdraw()
    filename = askopenfilename()
    root.destroy()
    return filename


if __name__ == '__main__':
    # Sanity check
    dataset = DataLoader(filebrowser())
    print('next():', dataset.next())
    print('next():', dataset.next())
    print('next():', dataset.next())
    print('prev():', dataset.prev())
    print('next():', dataset.next())

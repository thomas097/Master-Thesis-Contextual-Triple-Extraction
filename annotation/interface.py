import tkinter as tk
import tkinter.font as TkFont
from dataloader import DataLoader
from functools import partial


## Widgets

class TripleButton(tk.Button):
    def __init__(self, root, command, relief='groove'):
        # Init Button
        self._var = tk.StringVar()
        super().__init__(root, textvariable=self._var, relief=relief, command=command)
        self.pack(side=tk.LEFT)

        # Register assigned tokens
        self._root = root
        self.tokens = []
        self.indices = []
        self.clear()

    def highlight(self, value, color='lightgrey'):
        """ Highlight the button when selected.
        """
        if value:
            self.configure(bg=color)
        else:
            self.configure(bg=self._root.cget('bg'))

    def add(self, token, index):
        """ Assigns token to argument.
        """
        self.tokens.append(token)
        self.indices.append(index)
        self._var.set(' '.join(self.tokens))

    def clear(self, minsize=8):
        """ Erase all tokens of button.
        """
        self.tokens = []
        self.indices = []
        self._var.set(' ' * minsize)


class Row(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.pack(side=tk.TOP)


class Column(tk.Frame):
    def __init__(self, root, row=0, col=0, colspan=1):
        super().__init__(root)
        self.grid(row=row, column=col, columnspan=colspan, padx=8, pady=2)
        self._rows = []

    def _expand(self, num):
        """ Add rows until there are 'num'
        """
        while num > len(self._rows) - 1:
            self._rows.append(Row(self))

    def add_button(self, i, text, command):
        self._expand(i)
        button = tk.Button(self._rows[i], text=text, relief='groove', command=command)
        button.pack(side=tk.LEFT)
        return button

    def add_text(self, i, text, pad=None):
        self._expand(i)
        label = tk.Label(self._rows[i], text=text, relief='flat')
        if pad is not None:
            label.pack(side=tk.LEFT, padx=pad)
        else:
            label.pack(side=tk.LEFT)
        return label

    def add_triple(self, i, command):
        self._expand(i)
        self.add_text(i, '⧼')
        subj = TripleButton(self._rows[i], relief='groove', command=partial(command, i, 0))
        self.add_text(i, ',')
        pred = TripleButton(self._rows[i], relief='groove', command=partial(command, i, 1))
        self.add_text(i, ',')
        obj = TripleButton(self._rows[i], relief='groove', command=partial(command, i, 2))
        self.add_text(i, '⧽')
        return subj, pred, obj

    def add_perspective(self, i, command):
        self._expand(i)
        self.add_text(i, '⧼')
        polarity = TripleButton(self._rows[i], relief='groove', command=partial(command, i, 3))
        self.add_text(i, ',')
        certainty = TripleButton(self._rows[i], relief='groove', command=partial(command, i, 4))
        self.add_text(i, '⧽')
        return polarity, certainty

    def add_padding(self, n):
        start = len(self._rows)
        for i in range(start, start + n):
            self.add_text(i, '', pad=20)


## Interface

class Interface:
    def __init__(self, dataloader, num_triples=5, title='Annotation Tool', fontsize=15):
        # Window
        self._root = tk.Tk()
        self._root.title(title)
        self._font = TkFont.nametofont("TkDefaultFont")
        self._font.configure(size=fontsize)
        self._root.option_add("*Font", self._font)
        self._title = title

        # Create Dialog, Triple and Perspective columns
        self._dialog_col = Column(self._root, row=0, col=0, colspan=2)
        self._triple_col = Column(self._root, row=1, col=0, colspan=1)
        self._persp_col = Column(self._root, row=1, col=1, colspan=1)
        self._button_col = Column(self._root, row=2, col=0, colspan=2)

        # Initial conditions
        self._dataloader = dataloader
        self._current = self._dataloader.next()

        self._num_triples = num_triples
        self._focus = (0, 0)  # What argument is currently being annotated?
        self._token_buttons = {}
        self._triple_buttons = {}

        # Start annotation loop
        self._annotate()
        self._root.mainloop()

    def _save_and_next(self):
        # Save annotation to file
        annotation = {'tokens': self._current, 'annotations': []}
        for i in range(self._num_triples):
            subj = self._triple_buttons[(i, 0)].indices
            pred = self._triple_buttons[(i, 1)].indices
            obj = self._triple_buttons[(i, 2)].indices
            polarity = self._triple_buttons[(i, 3)].indices
            certainty = self._triple_buttons[(i, 4)].indices
            annotation['annotations'].append((subj, pred, obj, polarity, certainty))

        self._dataloader.save(annotation)
        self._skip()

    def _clear(self):
        self._dialog_col.destroy()
        self._triple_col.destroy()
        self._persp_col.destroy()
        self._button_col.destroy()

        self._dialog_col = Column(self._root, row=0, col=0, colspan=2)
        self._triple_col = Column(self._root, row=1, col=0, colspan=1)
        self._persp_col = Column(self._root, row=1, col=1, colspan=1)
        self._button_col = Column(self._root, row=2, col=0, colspan=2)

        self._focus = (0, 0)
        self._token_buttons = {}
        self._triple_buttons = {}

    def _skip(self):
        self._clear()
        self._current = self._dataloader.next()
        self._annotate()

    def _go_back(self):
        self._clear()
        self._current = self._dataloader.prev()
        self._annotate()

    def _assign_to_focus(self, i, j):
        token = self._current[i][j]
        button = self._triple_buttons[self._focus]
        button.add(token, (i, j))

    def _set_focus(self, i, j):
        # Set focus to the j-th argument of the i-th triple
        self._focus = (i, j)

        #  Set color of selected argument
        for idx, button in self._triple_buttons.items():
            if (i, j) == idx:
                button.highlight(True)
                button.clear()
            else:
                button.highlight(False)

    def _annotate(self):
        # Populate dialog Column with Tokens
        for i, turn in enumerate(self._current):
            for j, token in enumerate(turn):
                token = self._dialog_col.add_button(2 * i, token, command=partial(self._assign_to_focus, i, j))
                self._token_buttons[(i, j)] = token
        self._dialog_col.add_padding(1)

        # Populate triple and perspective Columns
        for i in range(self._num_triples):
            subject, predicate, object_ = self._triple_col.add_triple(i, command=self._set_focus)
            self._triple_buttons[(i, 0)] = subject
            self._triple_buttons[(i, 1)] = predicate
            self._triple_buttons[(i, 2)] = object_

            polarity, certainty = self._persp_col.add_perspective(i, command=self._set_focus)
            self._triple_buttons[(i, 3)] = polarity
            self._triple_buttons[(i, 4)] = certainty
        self._triple_col.add_padding(1)

        # Add Skip and Next buttons
        self._button_col.add_button(1, 'Back', command=self._go_back)
        self._button_col.add_button(1, 'Skip', command=self._skip)
        self._button_col.add_button(1, 'Save + Next', command=self._save_and_next)

        # Set title to annotation id
        self._root.title(self._title + ": " + self._dataloader.current_id)


if __name__ == '__main__':
    dataloader = DataLoader('datasets/train.json', output_dir='newest_annotations')
    interface = Interface(dataloader)

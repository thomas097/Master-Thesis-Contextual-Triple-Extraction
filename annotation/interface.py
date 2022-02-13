import tkinter as tk
import tkinter.font as TkFont
from dataloader import DatasetIO
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
        self._tokens = []
        self._indices = []
        self.clear()

    @property
    def indices(self):
        return self._indices

    def highlight(self, value):
        if value:
            self.configure(bg='lightgrey')
        else:
            self.configure(bg=self._root.cget('bg'))

    def add(self, token, index):
        self._tokens.append(token)
        self._indices.append(index)
        self._var.set(' '.join(self._tokens))

    def clear(self):
        self._tokens = []
        self._indices = []
        self._var.set(' ' * 8)


class PerspectiveCheckbox(tk.Checkbutton):
    def __init__(self, root):
        self._var = tk.IntVar()
        super().__init__(root, variable=self._var, onvalue=0, offvalue=1)  # unchecked = positive
        self.pack(side=tk.LEFT)
        self.deselect()

    def set(self, polarity):
        if polarity:
            self.deselect()  # unchecked = positive
        else:
            self.select()

    @property
    def polarity(self):
        return self._var.get()


class Entry(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.pack(side=tk.LEFT)


class Row(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.pack(side=tk.TOP)


class Column(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.pack(side=tk.LEFT, padx=8, pady=16)
        self._rows = []

    def _expand(self, i):
        # Add rows until there are enough
        while i > len(self._rows) - 1:
            self._rows.append(Row(self))

    def add_button(self, i, text, command):
        self._expand(i)
        button = tk.Button(self._rows[i], text=text, relief='groove', command=command)
        button.pack(side=tk.LEFT)
        return button

    def add_text(self, i, text):
        self._expand(i)
        label = tk.Label(self._rows[i], text=text, relief='flat')
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

    def add_toggle(self, i, text):
        self._expand(i)
        toggle = PerspectiveCheckbox(self._rows[i])
        self.add_text(i, text)
        return toggle

    def add_padding(self, n):
        start = len(self._rows)
        for i in range(start, start + n):
            self.add_text(i, '')


## Interface

class Interface:
    def __init__(self, dataloader, parser=None, title='Annotation Tool', fontsize=15):
        # Window
        self._root = tk.Tk()
        self._root.title(title)
        self._font = TkFont.nametofont("TkDefaultFont")
        self._font.configure(size=fontsize)
        self._root.option_add("*Font", self._font)

        # Create Dialog, Triple and Perspective columns
        self._dialog_col = Column(self._root)
        self._triple_col = Column(self._root)
        self._persp_col = Column(self._root)

        # Initial conditions
        self._parser = parser
        self._dataloader = dataloader
        self._current = self._dataloader.next()
        if self._current is None:
            quit()

        self._triple_focus = (0, 0)
        self._tokens = {}
        self._triples = {}
        self._perspectives = {}

        # Start annotation loop
        self._annotate(self._current)
        self._root.mainloop()

    def _next_sample(self):
        # Save annotation to file
        annotation = {'tokens': self._current, 'triples': [], 'perspectives': []}
        for i in range(len(self._current)):
            subj = self._triples[(i, 0)].indices
            pred = self._triples[(i, 1)].indices
            obj = self._triples[(i, 2)].indices
            polarity = self._perspectives[i].polarity
            annotation['triples'].append((subj, pred, obj))
            annotation['perspectives'].append({'polarity': polarity})

        self._dataloader.save(annotation)

        # Remove old columns
        self._dialog_col.destroy()
        self._triple_col.destroy()
        self._persp_col.destroy()
        self._triple_focus = (0, 0)
        self._tokens = {}
        self._triples = {}
        self._perspectives = {}

        # Load new data
        self._dialog_col = Column(self._root)
        self._triple_col = Column(self._root)
        self._persp_col = Column(self._root)
        self._current = self._dataloader.next()

        if self._current is None:
            quit()

        self._annotate(self._current)

    def _assign_to_triple(self, i, j):
        token = self._current[i][j]
        button = self._triples[self._triple_focus]
        button.add(token, (i, j))

    def _set_focus(self, i, j):
        # Set focus to the j-th argument of the i-th triple
        self._triple_focus = (i, j)

        #  Set color of selected argument
        for idx, button in self._triples.items():
            if (i, j) == idx:
                button.highlight(True)
                button.clear()
            else:
                button.highlight(False)

    def _annotate(self, dialog):
        # Populate dialog Column with Tokens
        for i, turn in enumerate(dialog):
            for j, token in enumerate(turn):
                token = self._dialog_col.add_button(i, token, command=partial(self._assign_to_triple, i, j))
                self._tokens[(i, j)] = token
        self._dialog_col.add_padding(2)

        # Populate triple and perspective Columns
        for i, turn in enumerate(dialog):
            triple = self._triple_col.add_triple(i, command=self._set_focus)
            for j in range(3):
                self._triples[(i, j)] = triple[j]  # e.g. Button of Subject

            self._perspectives[i] = self._persp_col.add_toggle(i, 'Negated?')

        self._triple_col.add_padding(1)
        self._persp_col.add_padding(1)

        # If parser was given, pre-populate the buttons
        if self._parser is not None:
            for turn, (triple, persp) in enumerate(zip(*self._parser.parse(dialog))):
                # Set each argument one-by-one
                for arg in range(3):
                    for i, j in triple[arg]:  # Loop through token idx in arg
                        token = dialog[i][j]
                        self._triples[(turn, arg)].add(token, (i, j))

                # Set perspective
                self._perspectives[turn].set(persp)

        # Add Next button
        self._persp_col.add_button(len(dialog) + 1, 'Next', command=self._next_sample)


if __name__ == '__main__':
    dataloader = DatasetIO('data.txt')
    interface = Interface(dataloader)

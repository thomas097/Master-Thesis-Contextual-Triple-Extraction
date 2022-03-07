import tkinter as tk
import tkinter.font as TkFont
from dataloader import DataLoader
from functools import partial

from config import *


class ArgumentButton(tk.Button):
    def __init__(self, root, command, align=tk.LEFT):
        # Create Button
        self._label = tk.StringVar()
        super().__init__(root, textvariable=self._label, relief=ARG_RELIEF, bg=BG_COLOR, command=command)
        self.pack(side=align)

        # Register assigned tokens
        self._root = root
        self.tokens = []
        self.indices = []
        self.clear()

    def set_placeholder(self, string):
        """ Sets a placeholder with a text description
        """
        self.config(fg=ARG_PLACEHOLDER_TEXT_COLOR)
        self._label.set(string)

    def highlight(self, value):
        """ Highlight the button when selected.
        """
        if value:
            self.configure(bg=ARG_HIGHLIGHT_COLOR)
        else:
            self.configure(bg=BG_COLOR)

    def add(self, token, index):
        """ Assigns token to argument.
        """
        self.tokens.append(token)
        self.indices.append(index)
        self.config(fg=ARG_FONT_COLOR)
        self._label.set(' '.join(self.tokens))

    def clear(self):
        """ Erase all tokens of button.
        """
        self.tokens = []
        self.indices = []
        self._label.set(ARG_PLACEHOLDER)


class Row(tk.Frame):
    def __init__(self, root):
        super().__init__(root, bg=BG_COLOR)
        self.pack(side=tk.TOP)


class Column(tk.Frame):
    def __init__(self, root, row=0, col=0, colspan=1, sticky=tk.E):
        super().__init__(root, bg=BG_COLOR)
        self.grid(row=row, column=col, columnspan=colspan, padx=COLUMN_PADDING, pady=COLUMN_PADDING, sticky=sticky)
        self._rows = []

    def _expand(self, num):
        """ Add rows until there are 'num'
        """
        while num > len(self._rows) - 1:
            self._rows.append(Row(self))

    def add_button(self, i, text, command, font_color=FONT_COLOR, relief=RELIEF, bg_color=BG_COLOR, padding=PADDING):
        self._expand(i)
        button = tk.Button(self._rows[i], text=text, relief=relief, command=command, bg=bg_color, fg=font_color)
        button.pack(side=tk.LEFT, padx=padding, pady=padding, ipadx=padding, ipady=padding)
        return button

    def add_text(self, i, text, pad=0, align=tk.LEFT, bg_color=BG_COLOR):
        self._expand(i)
        label = tk.Label(self._rows[i], text=text, relief='flat', bg=bg_color)
        label.pack(side=align, padx=pad)
        return label

    def add_triple(self, i, command):
        self._expand(i)
        self.add_text(i, '⧼')
        subj = ArgumentButton(self._rows[i], command=partial(command, i, 0))
        self.add_text(i, ',')
        pred = ArgumentButton(self._rows[i], command=partial(command, i, 1))
        self.add_text(i, ',')
        obj = ArgumentButton(self._rows[i], command=partial(command, i, 2))
        self.add_text(i, '⧽')
        return subj, pred, obj

    def add_perspective(self, i, command):
        self._expand(i)
        self.add_text(i, '⧼')
        polarity = ArgumentButton(self._rows[i], command=partial(command, i, 3))
        self.add_text(i, ',')
        certainty = ArgumentButton(self._rows[i], command=partial(command, i, 4))
        self.add_text(i, '⧽')
        return polarity, certainty


class Interface:
    def __init__(self, dataloader):
        # Window
        self._window = tk.Tk()
        self._window.title(TITLE)
        tkfont = TkFont.nametofont(FONT_STYLE)
        tkfont.configure(size=FONT_SIZE)
        self._window.option_add("*Font", tkfont)

        # Centered Frame to add interface into
        self._root = tk.Frame(self._window, bg=BG_COLOR)
        self._root.pack(fill="none", expand=True)

        # Layout
        self._token_frame = Column(self._root, row=0, col=0, colspan=2, sticky=tk.N)
        self._triple_frame = Column(self._root, row=1, col=0, colspan=1, sticky=tk.E)
        self._persp_frame = Column(self._root, row=1, col=1, colspan=1, sticky=tk.W)
        self._button_frame = Column(self._root, row=2, col=0, colspan=2, sticky=tk.S)
        self._tokens = {}
        self._triples = {}
        self._focus = None  # What argument is currently being annotated?

        # Bind L/R keys to change focus
        self._window.bind(LEFT_KEY, partial(self._change_focus_with_keys, 'l'))
        self._window.bind(RIGHT_KEY, partial(self._change_focus_with_keys, 'r'))

        # Annotate first sample
        self._dataloader = dataloader
        self._current = self._dataloader.next()

        self._init_layout()
        self._root.mainloop()

    def _change_focus_with_keys(self, evt, _):
        # Update focus position according to keys
        i, j = self._focus
        j = j - 1 if evt == 'l' else j + 1  # Change position in row

        if j < 0:
            i -= 1
            j = 4 # index of 5th argument
        elif j > 4:
            j = 0
            i += 1

        # Check if out of bounds
        if (i, j) in self._triples:
            self._set_focus(i, j)

    def _set_focus(self, i, j):
        # Set focus to the j-th argument of the i-th triple
        self._focus = (i, j)

        #  Set color of selected argument
        for idx, button in self._triples.items():
            if (i, j) == idx:
                button.highlight(True)
                button.clear()
            else:
                button.highlight(False)

    def _assign_to_focus(self, i, j):
        token = self._current[i][j]
        button = self._triples[self._focus]
        button.add(token, (i, j))

    def _save_and_next(self):
        # Save annotation to file
        annotation = {'tokens': self._current, 'annotations': []}
        for i in range(NUM_TRIPLES):
            triple = (self._triples[(i, 0)].indices,  # subject
                      self._triples[(i, 1)].indices,  # predicate
                      self._triples[(i, 2)].indices,  # object
                      self._triples[(i, 3)].indices,  # polarity
                      self._triples[(i, 4)].indices)  # certainty
            annotation['annotations'].append(triple)

        self._dataloader.save(annotation)
        self._skip()

    def _skip(self):
        self._current = self._dataloader.next()
        self._init_layout()

    def _go_back(self):
        self._current = self._dataloader.prev()
        self._init_layout()

    def _clear_layout(self):
        """ Clears all Columns and references to old Buttons
        """
        self._token_frame.destroy()
        self._triple_frame.destroy()
        self._persp_frame.destroy()
        self._button_frame.destroy()

        self._token_frame = None
        self._triple_frame = None
        self._persp_frame = None
        self._button_frame = None

        for button in self._tokens.values():
            button.destroy()

        for button in self._triples.values():
            button.destroy()

        self._tokens = {}
        self._triples = {}

    def _init_layout(self):
        # Add new Columns
        self._clear_layout()
        self._token_frame = Column(self._root, row=0, col=0, colspan=2, sticky=tk.N)
        self._triple_frame = Column(self._root, row=1, col=0, colspan=1, sticky=tk.E)
        self._persp_frame = Column(self._root, row=1, col=1, colspan=1, sticky=tk.W)
        self._button_frame = Column(self._root, row=2, col=0, colspan=2, sticky=tk.S)

        # Populate token Column
        for i, turn in enumerate(self._current):
            for j, token in enumerate(turn):
                token = self._token_frame.add_button(i, token, padding=TOKEN_PADDING, command=partial(self._assign_to_focus, i, j))
                self._tokens[(i, j)] = token

        # Populate triple and perspective Columns
        for i in range(NUM_TRIPLES):
            subject, predicate, object_ = self._triple_frame.add_triple(i, command=self._set_focus)
            self._triples[(i, 0)] = subject
            self._triples[(i, 1)] = predicate
            self._triples[(i, 2)] = object_

            polarity, certainty = self._persp_frame.add_perspective(i, command=self._set_focus)
            self._triples[(i, 3)] = polarity
            self._triples[(i, 4)] = certainty

        # Add Skip and Next buttons
        self._button_frame.add_button(1, ' ⮜ ', command=self._go_back, padding=BUTTON_PADDING,
                                      relief='flat', font_color=BUTTON_FONT_COLOR, bg_color=BACK_COLOR)
        self._button_frame.add_button(1, ' ✖ ', command=self._skip, padding=BUTTON_PADDING,
                                      relief='flat', font_color=BUTTON_FONT_COLOR, bg_color=SKIP_COLOR)
        self._button_frame.add_button(1, ' ✔ ', command=self._save_and_next, padding=BUTTON_PADDING,
                                      relief='flat', font_color=BUTTON_FONT_COLOR, bg_color=NEXT_COLOR)

        # Set default focus and placeholder text
        self._window.title('Annotating ' + self._dataloader.current_id)
        self._set_focus(0, 0)
        self._triples[(0, 0)].set_placeholder('  S  ')
        self._triples[(0, 1)].set_placeholder('  P  ')
        self._triples[(0, 2)].set_placeholder('  O  ')
        self._triples[(0, 3)].set_placeholder('  ¬  ')
        self._triples[(0, 4)].set_placeholder('  ?  ')


if __name__ == '__main__':
    dataloader = DataLoader('datasets/train.json', output_dir='annotations')
    interface = Interface(dataloader)

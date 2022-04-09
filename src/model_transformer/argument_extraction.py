import glob
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
from datetime import date

from transformers import logging
logging.set_verbosity(40)  # only errors

from utils import *


class ArgumentExtraction(torch.nn.Module):
    def __init__(self, base_model='albert-base-v2', path=None, output_dim=3, sep='<eos>'):
        """ Init a transformer with custom multi-span-extraction heads for SPO arguments
        """
        super().__init__()
        # Base model
        print('loading %s for argument extraction' % base_model)
        # Load base model
        self._model = AutoModel.from_pretrained(base_model)
        self._base = base_model
        self._sep = sep

        # Load and extend tokenizer with SPEAKERS
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        self._tokenizer.add_tokens(['SPEAKER1', 'SPEAKER2'], special_tokens=True)
        self._model.resize_token_embeddings(len(self._tokenizer))

        # BIO classification heads
        hidden_size = AutoConfig.from_pretrained(base_model).hidden_size
        self._subj_head = torch.nn.Linear(hidden_size, output_dim)
        self._pred_head = torch.nn.Linear(hidden_size, output_dim)
        self._obj_head = torch.nn.Linear(hidden_size, output_dim)
        self._output_dim = output_dim

        self._relu = torch.nn.ReLU()
        self._softmax = torch.nn.Softmax(dim=-1)

        # Enable support for GPU (optional)
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

        # Load model / tokenizer if pretrained model is given
        if path:
            print('\t- Loading pretrained')
            model_path = glob.glob(path + '/argument_extraction_*')[0]
            self.load_state_dict(torch.load(model_path, map_location=self._device))

    def forward(self, input_ids, speaker_ids):
        """ Computes the forward pass through the model
        """
        # Feed dialog through transformer
        y = self._model(input_ids=input_ids, token_type_ids=speaker_ids)
        h = self._relu(y.last_hidden_state)

        # Predict spans
        y_subj = self._softmax(self._subj_head(h))
        y_pred = self._softmax(self._pred_head(h))
        y_obj_ = self._softmax(self._obj_head(h))

        # Permute output as tensor of shape (N, |C|, seq_len)
        y_subj = y_subj.permute(0, 2, 1)
        y_pred = y_pred.permute(0, 2, 1)
        y_obj_ = y_obj_.permute(0, 2, 1)
        return y_subj, y_pred, y_obj_

    def _retokenize_tokens(self, tokens):
        # Tokenize each token individually (keeping track of subwords)
        input_ids = [[self._tokenizer.cls_token_id]]
        for t in tokens:
            if t != '<eos>':
                input_ids.append(self._tokenizer.encode(t, add_special_tokens=False))
            else:
                input_ids.append([self._tokenizer.eos_token_id])

        # Flatten input_ids
        f_input_ids = torch.LongTensor([[i for ids in input_ids for i in ids]]).to(self._device)

        # Determine how often we need to repeat the labels
        repeats = [len(ids) for ids in input_ids]

        # Determine speaker ids (0 or 1)
        speaker_ids = [0] + [tokens[:i + 1].count(self._sep) % 2 for i in range(len(tokens))][:-1]  # TODO: make pretty
        speaker_ids = self._repeat_speaker_ids(speaker_ids, repeats)

        return f_input_ids, speaker_ids, repeats

    def _repeat_speaker_ids(self, speaker_ids, repeats):
        rep_speaker_ids = np.repeat([0] + list(speaker_ids), repeats=repeats)
        return torch.LongTensor([rep_speaker_ids]).to(self._device)

    def _repeat_labels(self, labels, repeats):
        # Repeat each label b the amount of subwords per token
        rep_labels = []
        for label, rep in zip([0] + list(labels), repeats):
            if label == 0:  # Outside
                rep_labels += [label] * rep
            else:  # Beginning + Inside
                rep_labels += [label] + ([2] * (rep - 1))  # If label = B -> B-I-I-I...
        return torch.LongTensor([rep_labels]).to(self._device)

    def fit(self, tokens, labels, epochs=2, lr=1e-5, weight=3):
        """ Fits the model to the annotations
        """
        # Re-tokenize to obtain input_ids and associated labels
        X = []
        for token_seq, (subj_labels, pred_labels, _obj_labels) in zip(tokens, labels):
            input_ids, speaker_ids, repeats = self._retokenize_tokens(token_seq)
            subj_labels = self._repeat_labels(subj_labels, repeats)  # repeat when split into subwords
            pred_labels = self._repeat_labels(pred_labels, repeats)
            _obj_labels = self._repeat_labels(_obj_labels, repeats)
            X.append((input_ids, speaker_ids, subj_labels, pred_labels, _obj_labels))

        # Set up optimizer
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        # Higher weight for B- and I-tags to account for class imbalance
        class_weights = torch.Tensor([1] + [weight] * (self._output_dim - 1)).to(self._device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        print('Training!')
        for epoch in range(epochs):
            losses = []
            random.shuffle(X)
            for input_ids, speaker_ids, subj_y, pred_y, obj_y in tqdm(X):
                # Forward pass
                subj_y_hat, pred_y_hat, obj_y_hat = self(input_ids, speaker_ids)

                loss = criterion(subj_y_hat, subj_y)  # Subj error
                loss += criterion(pred_y_hat, pred_y)  # Pred error
                loss += criterion(obj_y_hat, obj_y)  # Obj error
                losses.append(loss.item())

                optim.zero_grad()
                loss.backward()
                optim.step()

            print("mean loss =", np.mean(losses))

        # Save model to file
        torch.save(self.state_dict(), 'argument_extraction_%s_%s' % (self._base, date.today()))

    def predict(self, token_seq):
        # Retokenize token sequence
        input_ids, speaker_ids, _ = self._retokenize_tokens(token_seq)

        # Invert tokenization for viewing
        subwords = self._tokenizer.convert_ids_to_tokens(input_ids[0])

        # Forward-pass
        subj_y_hat, pred_y_hat, _obj_y_hat = self(input_ids, speaker_ids)
        subj = subj_y_hat.cpu().detach().numpy()[0]
        pred = pred_y_hat.cpu().detach().numpy()[0]
        _obj = _obj_y_hat.cpu().detach().numpy()[0]
        return subj, pred, _obj, subwords


if __name__ == '__main__':
    annotations = load_annotations('<path_to_annotation_file')

    # Convert argument annotations to BIO-tag sequences
    tokens, labels = [], []
    for ann in annotations:
        # Map triple arguments to BIO tagged masks
        labels.append((triple_to_bio_tags(ann, 0),
                       triple_to_bio_tags(ann, 1),
                       triple_to_bio_tags(ann, 2)))

        # Flatten turn sequence
        tokens.append([t for ts in ann['tokens'] for t in ts + ['<eos>']])

    # Fit model to data
    model = ArgumentExtraction()
    model.fit(tokens, labels)
    torch.save(model.state_dict(), 'models/argument_extraction_albert-v2_31_03_2022')


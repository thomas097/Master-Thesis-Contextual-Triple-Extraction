import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

from transformers import logging
logging.set_verbosity(40)  # only errors

from utils import *


class TripleScoring(torch.nn.Module):
    def __init__(self, base_model='albert-base-v2', path=None):
        super().__init__()
        # Base model
        print('loading', base_model)
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        self._model = AutoModel.from_pretrained(base_model)

        # SPO candidate scoring heads
        hidden_size = AutoConfig.from_pretrained(base_model).hidden_size
        self._entailed_head = torch.nn.Linear(hidden_size, 2)
        self._polarity_head = torch.nn.Linear(hidden_size, 2)

        self._relu = torch.nn.ReLU()
        self._softmax = torch.nn.Softmax(dim=-1)

        # GPU support
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

        if path is not None:
            self.load_state_dict(torch.load(path, map_location=self._device))

    def forward(self, input_ids, speaker_ids):
        """ Computes the forward pass through the model
        """
        # Feed dialog through transformer
        out = self._model(input_ids=input_ids, token_type_ids=speaker_ids)
        h = self._relu(out.last_hidden_state[:, 0])

        # Predict spans
        y_ent = self._softmax(self._entailed_head(h))
        y_pol = self._softmax(self._polarity_head(h))
        return y_ent, y_pol

    def _retokenize_tokens(self, tokens, triple, speaker=0):
        # Tokenize each token individually (keeping track of subwords)
        f_input_ids = [self._tokenizer.cls_token_id]
        speaker_ids = [0]
        for t in tokens:
            if t != '<eos>':
                token_ids = self._tokenizer.encode(' ' + t, add_special_tokens=False)
                f_input_ids += token_ids
                speaker_ids += [speaker] * len(token_ids)  # repeat speaker_id if subword tokenized
            else:
                f_input_ids.append(self._tokenizer.eos_token_id)
                speaker_ids.append(speaker)
                speaker = 1 - speaker

        # Add [PAD] as spacer
        f_input_ids[-1] = self._tokenizer.pad_token_id

        # Append triple
        triple_ids = self._tokenizer.encode(' '.join(triple), add_special_tokens=False)
        f_input_ids += triple_ids
        speaker_ids += [0] * len(triple_ids)

        f_input_ids = torch.LongTensor([f_input_ids]).to(self._device)
        speaker_ids = torch.LongTensor([speaker_ids]).to(self._device)
        return f_input_ids, speaker_ids

    def fit(self, tokens, triples, labels, epochs=2, lr=1e-6):
        """ Fits the model to the annotations
        """
        X = []
        for tokens, triple_lst, (ent_labels, pol_labels) in zip(tokens, triples, labels):
            for triple, entailed, polarity in zip(triple_lst, ent_labels, pol_labels):
                # Put data on GPU
                input_ids, speaker_ids = self._retokenize_tokens(tokens, triple)
                entailed = torch.LongTensor([entailed]).to(self._device)
                polarity = torch.LongTensor([polarity]).to(self._device) if polarity is not None else None

                X.append((input_ids, speaker_ids, entailed, polarity))

        # Set up optimizer and objective
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            random.shuffle(X)

            losses = []
            for input_ids, speaker_ids, y_ent, y_pol in tqdm(X):

                # Update w.r.t. entailment
                y_ent_hat, _ = self(input_ids, speaker_ids)
                loss = criterion(y_ent_hat, y_ent)  # Was the triple correct?
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update w.r.t. polarity
                if y_pol is not None:
                    _, y_pol_hat = self(input_ids, speaker_ids)
                    loss = criterion(y_pol_hat, y_pol)  # What was the polarity?
                    losses.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            print("mean loss =", np.mean(losses))

    def predict(self, tokens, triple):
        input_ids, speaker_ids = self._retokenize_tokens(tokens, triple)
        entailed, polarity = self(input_ids, speaker_ids)
        entailed = entailed.cpu().detach().numpy()[0]
        polarity = polarity.cpu().detach().numpy()[0]
        return entailed, polarity


if __name__ == '__main__':
    annotations = load_annotations('<path_to_annotation_file')

    # Extract annotation triples and compute negative triples
    tokens, triples, labels = [], [], []
    for ann in annotations:
        ann_triples, ann_ent, ann_pol = extract_triples(ann)
        triples.append(ann_triples)
        labels.append([ann_ent, ann_pol])
        tokens.append([t for ts in ann['tokens'] for t in ts + ['<eos>']])

    # Fit model
    scorer = TripleScoring()
    scorer.fit(tokens, triples, labels)
    torch.save(scorer.state_dict(), 'models/scorer_albert-v2_03_03_2022')



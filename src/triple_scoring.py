import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

from transformers import logging
logging.set_verbosity(40)  # only errors

from utils import *


class TripleScoring(torch.nn.Module):
    def __init__(self, base_model='albert-base-v2', path=None):
        super().__init__()
        print('loading', base_model)
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        self._model = AutoModel.from_pretrained(base_model)

        # SPO extraction heads
        config = AutoConfig.from_pretrained(base_model)
        self._entailment_head = torch.nn.Linear(config.hidden_size, 2)
        self._polarity_head = torch.nn.Linear(config.hidden_size, 2)

        self._relu = torch.nn.ReLU()
        self._softmax = torch.nn.Softmax(dim=-1)

        # GPU support
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

        if path is not None:
            self.load_state_dict(torch.load(path, map_location=self._device))

    def forward(self, x):
        """ Computes the forward pass through the model
        """
        h = self._model(x).last_hidden_state
        z = self._relu(h[:, 0])  # attach classifier at <s> token
        y0 = self._softmax(self._entailment_head(z))
        y1 = self._softmax(self._polarity_head(z))
        return y0, y1

    def _retokenize(self, tokens, triple):
        """ Takes in a sequence of turns and the triple and generates a list of
            input ids of the form: <s> turn_ids </s> ... </s> turn_ids <pad> triple_ids
            <pad> is used to indicate the triple
        """
        input_ids = [self._tokenizer.cls_token_id]
        for i, token in enumerate(tokens):
            if token == '<eos>':
                input_ids += [self._tokenizer.sep_token_id]
            else:
                if i == 0:
                    input_ids += self._tokenizer.encode(token, add_special_tokens=False)
                else:
                    input_ids += self._tokenizer.encode(' ' + token, add_special_tokens=False)

        input_ids += [self._tokenizer.pad_token_id]
        input_ids += self._tokenizer.encode(' '.join(triple), add_special_tokens=False)
        return input_ids

    def _batch_tokenize(self, tokens, triples):
        # Create a batch of positive and negative examples
        inputs = [self._retokenize(tokens, triple) for triple in triples]

        # Pad input ids sequences of samples in batch if needed
        max_len = max([len(ids) for ids in inputs])
        inputs = [ids + [self._tokenizer.unk_token_id] * (max_len - len(ids)) for ids in inputs]

        return torch.LongTensor(inputs).to(self._device)

    def fit(self, dialogs, triples, entailment_labels, polarity_labels, epochs=2, lr=1e-5):
        """ Fits the model to the annotations
        """
        # Put data on GPU
        X = []
        for tokens, triple_lst, labels1, labels2 in zip(dialogs, triples, entailment_labels, polarity_labels):
            batch = self._batch_tokenize(tokens, triple_lst)
            entailed = torch.LongTensor(labels1).to(self._device)
            polarity = torch.LongTensor(labels2).to(self._device)
            X.append((batch, entailed, polarity))

        # Set up optimizer and objective
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            losses = []
            random.shuffle(X)

            for x, ent_true, pol_true in tqdm(X):
                # Predict polarity and entailment
                ent_pred, pol_pred = self(x)

                # Compute loss
                loss = criterion(ent_pred, ent_true)  # Was the triple correct?

                for i, v in enumerate(pol_true):  # Only add to loss when entailed
                    if v > -1e-3:
                        loss += criterion(pol_pred[i:i+1], pol_true[i:i+1]) / len(pol_true)  # Was the triple confirmed or denied?

                losses.append(loss.item())

                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("mean loss =", np.mean(losses))

    def predict(self, tokens, triple):
        """ Given a sequence of tokens representing the dialog, predicts the
            triple arguments
        """
        # Tokenization
        input_ids = self._batch_tokenize(tokens, [triple])

        # Invert tokenization for viewing
        new_tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0])

        # Predict logit tensor for input
        y0, y1 = self(input_ids)
        y0 = y0.cpu().detach().numpy()[0][1]
        y1 = y1.cpu().detach().numpy()[0][1] # 2nd index = pos class
        return y0, y1, new_tokens


if __name__ == '__main__':
    annotations = load_annotations('<path_to_annotation_file')

    # Extract annotation triples and compute negative triples
    tokens, triples, entailment_labels, polarity_labels = [], [], [], []
    for ann in annotations:
        ann_triples, ann_ent, ann_pol = extract_triples(ann)
        triples.append(ann_triples)
        entailment_labels.append(ann_ent)
        polarity_labels.append(ann_pol)
        tokens.append([t for ts in ann['tokens'] for t in ts + ['<eos>']])

    # Fit model
    scorer = TripleScoring()
    scorer.fit(tokens, triples, entailment_labels, polarity_labels)
    torch.save(scorer.state_dict(), 'models/scorer_albert-v2_03_03_2022')



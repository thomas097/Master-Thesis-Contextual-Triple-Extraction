import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

from transformers import logging
logging.set_verbosity(40)  # only errors

from utils import *


class TripleScoring(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        self._tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self._model = RobertaModel.from_pretrained("roberta-base")

        # SPO extraction heads
        self._head = torch.nn.Linear(768, 1)
        self._relu = torch.nn.ReLU()
        self._sigmoid = torch.nn.Sigmoid()

        # GPU support
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

        # Load model if enabled
        if path:
            self.load_state_dict(torch.load(path, map_location=self._device))

    def forward(self, x):
        """ Computes the forward pass through the model
        """
        h = self._model(**x).last_hidden_state
        z = self._relu(h[:, 0])
        return self._sigmoid(self._head(z))  # attach classifier at <s> token

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

    def _batch_tokenize(self, batch_tokens, batch_pos_triples, batch_neg_triples):
        # Create a batch of positive and negative examples
        inputs, labels = [], []
        for tokens, pos_triples, neg_triples in zip(batch_tokens, batch_pos_triples, batch_neg_triples):
            for triple in pos_triples:
                inputs.append(self._retokenize(tokens, triple))
                labels.append([1])
            for triple in neg_triples:
                inputs.append(self._retokenize(tokens, triple))
                labels.append([0])

        # Pad input ids sequences of samples in batch if needed
        max_len = max([len(ids) for ids in inputs])
        attn_masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in inputs]
        inputs = [ids + [self._tokenizer.unk_token_id] * (max_len - len(ids)) for ids in inputs]

        # Shuffle samples to eliminate order effects
        samples = list(zip(inputs, attn_masks, labels))
        random.shuffle(samples)
        inputs, attn_masks, labels = zip(*samples)

        # Push to GPU
        inputs = {'input_ids': torch.LongTensor(inputs).to(self._device),
                  'attention_mask': torch.LongTensor(attn_masks).to(self._device)}
        labels = torch.Tensor(labels).to(self._device)
        return inputs, labels

    def fit(self, tokens, pos_triples, neg_triples, epochs=2, lr=1e-5, batch_size=1):
        """ Fits the model to the annotations
        """
        # Put data on GPU
        X = []
        for i in range(0, len(tokens), batch_size):
            batch_tokens = tokens[i:i + batch_size]
            batch_pos_triples = pos_triples[i:i + batch_size]
            batch_neg_triples = neg_triples[i:i + batch_size]
            batch = self._batch_tokenize(batch_tokens, batch_pos_triples, batch_neg_triples)
            X.append(batch)

        # Set up optimizer and objective
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()

        for epoch in range(epochs):
            losses = []
            random.shuffle(X)
            for x, y_true in tqdm(X):
                y_hat = self(x)

                loss = criterion(y_hat, y_true)  # Was the triple correct?
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("mean loss =", np.mean(losses))

    def predict(self, tokens, triple):
        """ Given a sequence of tokens representing the dialog, predicts the
            triple arguments
        """
        # Tokenization
        input_ids = self._retokenize(tokens, triple)
        inputs = {'input_ids': torch.LongTensor([input_ids]).to(self._device),
                  'attention_mask': torch.ones((1, len(input_ids))).to(self._device)}

        # Invert tokenization for viewing
        new_tokens = self._tokenizer.convert_ids_to_tokens(input_ids)

        # Predict logit tensor for input
        y = self(inputs).cpu().detach().numpy()
        return y[0], new_tokens


if __name__ == '__main__':
    annotations = load_annotations('<path_to_annotation_file')

    # Extract annotation triples and compute negative triples
    tokens, pos_triples, neg_triples = [], [], []
    for ann in annotations:
        triples = extract_triples(ann)
        pos_triples.append(triples)
        neg_triples.append(extract_negative_triples(triples))
        tokens.append([t for ts in ann['tokens'] for t in ts + ['<eos>']])

    # Fit model
    scorer = TripleScoring()
    scorer.fit(tokens, pos_triples, neg_triples)
    torch.save(scorer.state_dict(), 'models/scorer_model2.pt')



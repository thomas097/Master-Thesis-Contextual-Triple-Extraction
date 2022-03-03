import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

from transformers import logging
logging.set_verbosity(40)  # only errors

from utils import *


class ArgumentExtraction(torch.nn.Module):
    def __init__(self, base_model='albert-base-v2', path=None, output_dim=3):
        """ Inits a transformer with custom multi-span-extraction heads for SPO arguments
        """
        super().__init__()
        print('loading', base_model)
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        self._model = AutoModel.from_pretrained(base_model)

        config = AutoConfig.from_pretrained(base_model)
        self._subj_head = torch.nn.Linear(config.hidden_size, output_dim)
        self._pred_head = torch.nn.Linear(config.hidden_size, output_dim)
        self._obj_head = torch.nn.Linear(config.hidden_size, output_dim)
        self._output_dim = output_dim

        self._relu = torch.nn.ReLU()
        self._softmax = torch.nn.Softmax(dim=-1)

        # Enable support for GPU (optional)
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

        # Load model if enabled
        if path is not None:
            self.load_state_dict(torch.load(path, map_location=self._device))

    def forward(self, x):
        """ Computes the forward pass through the model
        """
        # Feed dialog through transformer
        z = self._relu(self._model(**x).last_hidden_state)

        # Predict argument spans as tensor: shape=(N, |C|, seq_len)
        y_subj = self._softmax(self._subj_head(z)).permute(0, 2, 1)
        y_pred = self._softmax(self._pred_head(z)).permute(0, 2, 1)
        y_obj = self._softmax(self._obj_head(z)).permute(0, 2, 1)
        return y_subj, y_pred, y_obj

    def _retokenize(self, tokens, old_labels=None):
        """ Retokenizes and relabels the token sequence in order to account
            for the subword tokenization used by RoBERTa.
        """
        # Start with <s>
        input_ids = [self._tokenizer.bos_token_id]
        labels = [0]

        # Tokenize each token separately
        for i, token in enumerate(tokens):

            # Add pad token
            if token == '<eos>':
                input_ids.append(self._tokenizer.sep_token_id)
                labels.append(0)

            else:
                token_ids = self._tokenizer.encode(' ' + token, add_special_tokens=False)
                input_ids += token_ids

                # Repeat label if len(token_ids) > 1
                if old_labels is not None:
                    labels += [old_labels[i]] * len(token_ids)

        # End with </s>
        input_ids[-1] = self._tokenizer.eos_token_id
        labels[-1] = 0

        # Convert to torch Tensors
        inputs = {'input_ids': torch.LongTensor([input_ids]).to(self._device),
                  'attention_mask': torch.ones(1, len(input_ids)).to(self._device)}
        outputs = torch.LongTensor([labels]).to(self._device)
        return inputs, outputs

    def fit(self, tokens, labels, epochs=2, lr=1e-5, weight=2):
        """ Fits the model to the annotations
        """
        # Re-tokenize tokens to obtain input_ids and SPO labels
        X = []
        for token_seq, label_seq in zip(tokens, labels):
            input_ids, subj_label = self._retokenize(token_seq, label_seq[0])
            _, pred_label = self._retokenize(token_seq, label_seq[1])
            _, obj_label = self._retokenize(token_seq, label_seq[2])
            X.append((input_ids, subj_label, pred_label, obj_label))

        # Set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # CCE with higher class weight for B and I tags (to account for class imbalance)
        class_weights = torch.Tensor([1] + [weight] * (self._output_dim - 1))
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(self._device))

        for epoch in range(epochs):
            losses = []
            random.shuffle(X)
            for x, y_subj, y_pred, y_obj in tqdm(X):
                y_subj_hat, y_pred_hat, y_obj_hat = self(x)

                loss = criterion(y_subj_hat, y_subj)  # subj error
                loss += criterion(y_pred_hat, y_pred)  # pred error
                loss += criterion(y_obj_hat, y_obj)  # obj error
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("mean loss =", np.mean(losses))

    def predict(self, tokens):
        """ Given a sequence of tokens representing the dialog, predicts the
            triple arguments
        """
        # Tokenization
        input_ids, _ = self._retokenize(tokens)

        # Invert tokenization for viewing
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids['input_ids'][0])

        # Predict logit tensor for input
        y_subj, y_pred, y_obj = self(input_ids)
        subjs = y_subj.cpu().detach().numpy()[0]
        preds = y_pred.cpu().detach().numpy()[0]
        objs = y_obj.cpu().detach().numpy()[0]

        return subjs, preds, objs, tokens


if __name__ == '__main__':
    annotations = load_annotations('<path_to_annotation_file')

    # Convert argument annotations to BIO-tag sequences
    tokens, labels = [], []
    for ann in annotations:
        labels.append((triple_to_bio_tags(ann, 0),
                       triple_to_bio_tags(ann, 1),
                       triple_to_bio_tags(ann, 2)))

        # Flatten dialogs to token sequence separated by <eos>
        tokens.append([t for ts in ann['tokens'] for t in ts + ['<eos>']])

    # Fit model to data
    model = ArgumentExtraction()
    model.fit(tokens, labels)
    torch.save(model.state_dict(), 'models/argument_extraction_albert-v2_03_03_2022')


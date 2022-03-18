import os
import json
import random


if __name__ == '__main__':
    SAMPLES_PER_BATCH = 50
    NUM_BATCHES = 10
    DATASET = 'train.json'

    # Create directory to store batches into
    if not os.path.exists('batches'):
        os.mkdir('batches')

    with open(DATASET, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Divide subset of N*K samples into K batches
    samples = random.sample(data, SAMPLES_PER_BATCH * NUM_BATCHES)
    for i in range(0, NUM_BATCHES):
        j = i * SAMPLES_PER_BATCH
        batch = samples[j:j + SAMPLES_PER_BATCH]

        with open('batches/batch_N%s_%s.json' % (SAMPLES_PER_BATCH, i), 'w') as file:
            json.dump(batch, file, indent=4)

import os
import json
import random

random.seed(2)


if __name__ == '__main__':
    SAMPLES_PER_BATCH = 100
    NUM_SHARED_SAMPLES = 20
    PILOT_BATCH = 20
    NUM_BATCHES = 20
    DATASET = 'merged_trainval_unannotated.json'

    # Create directory to store batches into
    if not os.path.exists('batches'):
        os.mkdir('batches')

    # Load full dataset
    with open(DATASET, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Divide subset of N * K samples into K batches
    batches = []
    samples = random.sample(data, SAMPLES_PER_BATCH * NUM_BATCHES)
    for i in range(NUM_BATCHES):
        j = i * SAMPLES_PER_BATCH
        batch = samples[j:j + SAMPLES_PER_BATCH]
        batches.append(batch)

    # Pair batches and have them share NUM_SHARED_SAMPLES
    used_ids = set()
    for i in range(0, NUM_BATCHES, 2):
        batch1 = batches[i]
        batch2 = batches[i + 1]
        batch2[:NUM_SHARED_SAMPLES] = batch1[:NUM_SHARED_SAMPLES] # b1 -> b2

        # Keep track of ids in batches
        for sample in batch1 + batch2:
            used_ids.add(sample['id'])

        # Save both to file
        with open('batches/batch_%s.json' % i, 'w') as file1:
            json.dump(batch1, file1, indent=4)

        with open('batches/batch_%s.json' % (i + 1), 'w') as file2:
            json.dump(batch2, file2, indent=4)

    print("{} samples divided over {} batches".format(len(used_ids), NUM_BATCHES))

    # Create batch will all remaining samples
    remaining_batch = [sample for sample in data if sample['id'] not in used_ids]

    pilot_batch = []
    if len(remaining_batch) > PILOT_BATCH:
        pilot_batch = remaining_batch[:PILOT_BATCH]
        remaining_batch = remaining_batch[PILOT_BATCH:]
    print("{} in pilot batch".format(len(pilot_batch)))
    print("{} remaining".format(len(remaining_batch)))

    with open('batches/pilot_batch.json', 'w') as file:
        json.dump(pilot_batch, file, indent=4)

    with open('batches/remaining_batch.json', 'w') as file:
        json.dump(remaining_batch, file, indent=4)

    # Some dataset statistics
    print('\n### Sampling statistics ###')
    dataset_balance = {'Circa':0, 'PersonaChat':0, 'DailyDialogs':0}
    for sample in data:
        if 'circa' in sample['id']:
            dataset_balance['Circa'] += 1
        if 'personachat' in sample['id']:
            dataset_balance['PersonaChat'] += 1
        if 'daily_dialogs' in sample['id']:
            dataset_balance['DailyDialogs'] += 1
    print(dataset_balance)





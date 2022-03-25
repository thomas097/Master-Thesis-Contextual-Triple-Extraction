import os
import json
import random

random.seed(2)


if __name__ == '__main__':
    SAMPLES_PER_BATCH = 200
    NUM_SHARED_SAMPLES = 25
    PILOT_BATCH = 20
    NUM_BATCHES = 20
    DATASET = 'merged_trainval_unannotated.json'

    # Create directory to store batches into
    if not os.path.exists('batches'):
        os.mkdir('batches')

    # Load full dataset
    with open(DATASET, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Remove duplicates (just in case)
    used_ids = set()
    new_data = []
    for sample in data:
        if sample['id'] not in used_ids:
            new_data.append(sample)
        used_ids.add(sample['id'])
    data = new_data

    # Divide subset of N * K samples into K batches
    batches = []
    samples = random.sample(data, SAMPLES_PER_BATCH * NUM_BATCHES)
    for i in range(NUM_BATCHES):
        j = i * SAMPLES_PER_BATCH
        batch = samples[j:j + SAMPLES_PER_BATCH]
        batches.append(batch)

    # Share a subset of dialogs
    start = SAMPLES_PER_BATCH // 2
    shared_batch = batches[0][start:start + NUM_SHARED_SAMPLES]  # share middle 25 of first batch
    used_ids = set()
    for i, batch in enumerate(batches):
        batch[start:start + NUM_SHARED_SAMPLES] = shared_batch

        # Update used_ids with samples already in batch
        for sample in batch:
            used_ids.add(sample['id'])

        # Save to file
        with open('batches/batch_%s.json' % i, 'w') as file1:
            json.dump(batch, file1, indent=4)

    print("{} samples divided over {} batches".format(len(used_ids), NUM_BATCHES))

    # Create batch will all remaining samples
    remaining_batch = [sample for sample in data if sample['id'] not in used_ids]
    random.shuffle(remaining_batch)

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





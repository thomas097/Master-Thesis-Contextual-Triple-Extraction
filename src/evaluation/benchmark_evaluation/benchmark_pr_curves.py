import glob
import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_pr_curves(files, model_names):
    plt.figure(figsize=(6, 4))
    plt.grid(c='lightgrey')
    
    for fname, model_name in zip(files, model_names):
        with open(fname, 'r', encoding='utf-8') as file:
            data = json.load(file)
            recall, precision = data['pr-curve']
            plt.plot(recall, precision, label=model_name)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16),
               ncol=3, fancybox=True, shadow=True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


def load_auc_bars(files, model_names, margin=0.25):
    plt.figure(figsize=(6, 4))
    plt.grid(c='lightgrey', zorder=0)
    
    for i, (fname, model_name) in enumerate(zip(files, model_names)):
        with open(fname, 'r', encoding='utf-8') as file:
            data = json.load(file)
            plt.bar(i + 1, data['auc'], label=model_name, zorder=3)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16),
               ncol=3, fancybox=True, shadow=True)
    plt.xticks(range(1, len(files) + 1), model_names)
    plt.ylabel('AUC')
    plt.xlim(margin, len(files) + 1 - margin)
    plt.show()


if __name__ == '__main__':
    files = ['results/test_full_albert_k=0.96.json',
             'results/test_full_spacy_k=0.0.json',
             'results/test_full_ollie_k=0.0.json',
             'results/test_full_stanford_k=0.0.json',
             'results/test_full_openie5_k=0.0.json',
             'results/test_full_stanovsky_k=0.0.json']
    
    model_names = ['ALBERT', 'Dependency', 'OLLIE', 'Stanford',
                   'OpenIE5', 'Stanovsky\net al.']
    
    load_pr_curves(files, model_names)
    load_auc_bars(files, model_names)

    

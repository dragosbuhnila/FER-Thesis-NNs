import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.config import OCCLUDED_TRAIN_VAL_SET_H5_PATH, OCCLUDED_TEST_SET_H5_PATH,\
                            ORIGINAL_TRAIN_VAL_SET_H5_PATH, ADELE_TEST_SET_H5_PATH
import h5py
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# DATASET = 'occluded'  # for labeling purposes only; not used in code logic
DATASET = 'original'

if DATASET == 'occluded':
    TRAIN_VAL_PATH = OCCLUDED_TRAIN_VAL_SET_H5_PATH
    TEST_PATH = OCCLUDED_TEST_SET_H5_PATH
elif DATASET == 'original':
    TRAIN_VAL_PATH = ORIGINAL_TRAIN_VAL_SET_H5_PATH
    TEST_PATH = ADELE_TEST_SET_H5_PATH



def decode_names(barr):
    return [b.decode('utf-8') if isinstance(b, (bytes, bytearray)) else str(b) for b in barr]

def ensure_output_dir():
    out = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(out, exist_ok=True)
    return out

def counts_for_categories(values, categories):
    vals = np.array(values)
    counts = []
    for c in categories:
        counts.append(int(np.sum(vals == c)))
    return counts

def plot_grouped_bar(categories, counts_a, counts_b, label_a, label_b, title, fname):
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(categories)*0.5),4))
    rects_a = ax.bar(x - width/2, counts_a, width, label=label_a)
    rects_b = ax.bar(x + width/2, counts_b, width, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_title(title)
    ax.legend()

    # annotate bars with counts
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{int(h)}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    autolabel(rects_a)
    autolabel(rects_b)

    plt.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

def plot_single_bar(categories, counts, title, fname):
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(max(8, len(categories)*0.5),4))
    rects = ax.bar(x, counts)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_title(title)

    # annotate bars with counts
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{int(h)}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

def load_trainval(path):
    with h5py.File(path, 'r') as f:
        class_names = decode_names(f['class_names'][:])
        y_train = f['y_train'][:]
        y_val = f['y_val'][:]
    return class_names, y_train, y_val

def load_test(path):
    with h5py.File(path, 'r') as f:
        class_names = decode_names(f['class_names'][:])
        y_test = f['y_test'][:]
    return class_names, y_test

def extract_subject_ids_from_test_h5(path):
    """
    Try to find filename-like datasets in the HDF5 and extract subject ids of form 'bsNNN'.
    Returns a sorted list of unique subject ids (e.g. ['bs001','bs003']) or an empty list if none found.
    """
    subj_set = set()
    pattern = re.compile(r'bs\d{3}', re.IGNORECASE)
    candidate_keys = []
    with h5py.File(path, 'r') as f:
        for k in f.keys():
            lk = k.lower()
            if any(x in lk for x in ('file', 'name', 'path', 'img', 'id')):
                candidate_keys.append(k)
        # also try all datasets if none matched
        if not candidate_keys:
            candidate_keys = list(f.keys())

        for k in candidate_keys:
            try:
                vals = f[k][:]
            except Exception:
                continue
            # flatten and decode possible byte strings
            flat = np.array(vals).astype(object).ravel()
            for v in flat:
                if isinstance(v, (bytes, bytearray)):
                    try:
                        s = v.decode('utf-8', errors='ignore')
                    except Exception:
                        continue
                else:
                    s = str(v)
                for m in pattern.findall(s):
                    subj_set.add(m.lower())
    return sorted(subj_set)



if __name__ == "__main__":
    outdir = ensure_output_dir()
    class_names_tv, y_train, y_val = load_trainval(TRAIN_VAL_PATH)
    class_names_test, y_test = load_test(TEST_PATH)

    # prefer class names from trainval; ensure consistent order
    class_names = class_names_tv

    # convert numeric labels to names
    y_train_labels = [class_names[int(i)] for i in y_train]
    y_val_labels = [class_names[int(i)] for i in y_val]
    y_test_labels = [class_names[int(i)] for i in y_test]

    train_counts = counts_for_categories(y_train_labels, class_names)
    val_counts = counts_for_categories(y_val_labels, class_names)
    test_counts = counts_for_categories(y_test_labels, class_names)

    plot_grouped_bar(class_names, train_counts, val_counts, 'train', 'val',
                     'GT Emotions (train vs val)', os.path.join(outdir, f'{DATASET}_trainval_gt_statistics.png'))

    # compute subjects in test set (look for 'bsNNN' in filename-like datasets)
    subject_ids = extract_subject_ids_from_test_h5(TEST_PATH)
    n_subjects = len(subject_ids)
    if n_subjects > 0:
        test_fname = os.path.join(outdir, f'{DATASET}_test_gt_statistics_{n_subjects}subjects.png')
    else:
        test_fname = os.path.join(outdir, f'{DATASET}_test_gt_statistics.png')

    plot_single_bar(class_names, test_counts, 'GT Emotions (test)', test_fname)

    print(os.path.join(outdir, f'{DATASET}_trainval_gt_statistics.png'))
    print(test_fname)
    if n_subjects > 0:
        print(f'Found {n_subjects} unique test subjects: {subject_ids}')

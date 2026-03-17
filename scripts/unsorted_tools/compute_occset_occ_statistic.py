import sys; import os;
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.config import OCCLUDED_TEST_SET_H5_PATH, OCCLUDED_TRAIN_VAL_SET_H5_PATH
# ...existing code...
import re
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET = 'occluded'  # for labeling purposes only; not used in code logic


def annotate_bars(ax, rects, fmt=int, yoffset=3, fontsize=8):
    """
    Annotate bar containers with their heights above each bar.
    """
    for rect in rects:
        h = rect.get_height()
        try:
            label = fmt(h)
        except Exception:
            label = str(h)
        ax.annotate(str(label),
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, yoffset), textcoords="offset points",
                    ha='center', va='bottom', fontsize=fontsize)


def decode_names(barr):
    return [b.decode('utf-8') if isinstance(b, (bytes, bytearray)) else str(b) for b in barr]


def load_trainval(h5_path):
    with h5py.File(h5_path, 'r') as f:
        class_names = decode_names(f['class_names'][:])
        mismatch_train = f['mismatch_train'][:]
        mismatch_val = f['mismatch_val'][:]
        occ_train = f['occ_train'][:]
        occ_val = f['occ_val'][:]
        pos_train = f['pos_or_neg_train'][:]
        pos_val = f['pos_or_neg_val'][:]
    return {
        'class_names': class_names,
        'mismatch_train': mismatch_train,
        'mismatch_val': mismatch_val,
        'occ_train': occ_train,
        'occ_val': occ_val,
        'pos_train': pos_train,
        'pos_val': pos_val
    }


def load_test(h5_path):
    # parse from paths field: ...__masked-positive-SAD_mismatch.png
    pattern = re.compile(r'masked-(positive|negative)-([A-Z]+)_(match|mismatch)', re.IGNORECASE)
    with h5py.File(h5_path, 'r') as f:
        paths = [p.decode('utf-8') if isinstance(p, (bytes, bytearray)) else str(p) for p in f['paths'][:]]
    posneg = []
    occ_emotion = []
    mismatch = []
    for p in paths:
        m = pattern.search(p)
        if m:
            posneg.append(1 if m.group(1).lower() == 'positive' else 0)
            occ_emotion.append(m.group(2).upper())
            mismatch.append(1 if m.group(3).lower() == 'mismatch' else 0)
        else:
            # fallback: mark unknowns
            posneg.append(None)
            occ_emotion.append('UNKNOWN')
            mismatch.append(None)
    return {
        'paths': paths,
        'posneg': np.array([0 if v is None else v for v in posneg]),
        'occ_emotion': np.array(occ_emotion),
        'mismatch': np.array([0 if v is None else v for v in mismatch])
    }


def ensure_output_dir():
    out = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(out, exist_ok=True)
    return out


def plot_grouped_bar(categories, counts_a, counts_b, label_a, label_b, title, fname):
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(categories)*0.5),4))
    ax.bar(x - width/2, counts_a, width, label=label_a)
    ax.bar(x + width/2, counts_b, width, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def plot_single_bar(categories, counts, title, fname):
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(max(6, len(categories)*0.5),4))
    ax.bar(x, counts)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def counts_for_categories(values, categories):
    # values may be numeric indices or string labels
    vals = np.array(values)
    counts = []
    for c in categories:
        counts.append(int(np.sum(vals == c)))
    return counts


def plot_trainval_grid(outdir, class_names, mm_train, mm_val,
                       occ_train_counts, occ_val_counts,
                       pos_train_counts, pos_val_counts,
                       comb_cats, comb_train_counts, comb_val_counts):
    fig = plt.figure(figsize=(18,10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.8], hspace=0.4, wspace=0.3)

    ax_em = fig.add_subplot(gs[0, 0])
    x = np.arange(len(class_names)); w = 0.35
    rects_em_train = ax_em.bar(x - w/2, occ_train_counts, w, label='train')
    rects_em_val = ax_em.bar(x + w/2, occ_val_counts, w, label='val')
    ax_em.set_xticks(x); ax_em.set_xticklabels(class_names, rotation=45, ha='right')
    ax_em.set_title('Occlusion emotion (train vs val)'); ax_em.legend()
    annotate_bars(ax_em, rects_em_train); annotate_bars(ax_em, rects_em_val)

    ax_pos = fig.add_subplot(gs[0, 1])
    cats_pos = ['negative', 'positive']; x2 = np.arange(len(cats_pos))
    rects_pos_train = ax_pos.bar(x2 - w/2, pos_train_counts, w, label='train')
    rects_pos_val = ax_pos.bar(x2 + w/2, pos_val_counts, w, label='val')
    ax_pos.set_xticks(x2); ax_pos.set_xticklabels(cats_pos, rotation=45, ha='right')
    ax_pos.set_title('Occlusion pos/neg (train vs val)'); ax_pos.legend()
    annotate_bars(ax_pos, rects_pos_train); annotate_bars(ax_pos, rects_pos_val)

    ax_mm = fig.add_subplot(gs[0, 2])
    cats_mm = ['match', 'mismatch']; x3 = np.arange(len(cats_mm))
    rects_mm_train = ax_mm.bar(x3 - w/2, mm_train, w, label='train')
    rects_mm_val = ax_mm.bar(x3 + w/2, mm_val, w, label='val')
    ax_mm.set_xticks(x3); ax_mm.set_xticklabels(cats_mm, rotation=45, ha='right')
    ax_mm.set_title('Match vs Mismatch (train vs val)'); ax_mm.legend()
    annotate_bars(ax_mm, rects_mm_train); annotate_bars(ax_mm, rects_mm_val)

    ax_comb = fig.add_subplot(gs[1, :])
    x4 = np.arange(len(comb_cats)); w2 = 0.4
    rects_comb_train = ax_comb.bar(x4 - w2/2, comb_train_counts, w2, label='train')
    rects_comb_val = ax_comb.bar(x4 + w2/2, comb_val_counts, w2, label='val')
    ax_comb.set_xticks(x4); ax_comb.set_xticklabels(comb_cats, rotation=90, ha='right')
    ax_comb.set_title('Combined occlusion (train vs val)'); ax_comb.legend()
    annotate_bars(ax_comb, rects_comb_train); annotate_bars(ax_comb, rects_comb_val)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f'{DATASET}_trainval_occlusions_statistics.png'))
    plt.close(fig)


def plot_test_grid(outdir, cats_em_test, occ_test_counts,
                   pos_test_counts, mm_test, all_comb_cats, comb_test_counts):
    fig = plt.figure(figsize=(18,10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.8], hspace=0.4, wspace=0.3)

    ax_em = fig.add_subplot(gs[0, 0])
    x = np.arange(len(cats_em_test))
    rects_em = ax_em.bar(x, occ_test_counts)
    ax_em.set_xticks(x); ax_em.set_xticklabels(cats_em_test, rotation=45, ha='right')
    ax_em.set_title('Occlusion emotion (test)')
    annotate_bars(ax_em, rects_em)

    ax_pos = fig.add_subplot(gs[0, 1])
    cats_pos = ['negative', 'positive']; x2 = np.arange(len(cats_pos))
    rects_pos = ax_pos.bar(x2, pos_test_counts)
    ax_pos.set_xticks(x2); ax_pos.set_xticklabels(cats_pos, rotation=45, ha='right')
    ax_pos.set_title('Occlusion pos/neg (test)')
    annotate_bars(ax_pos, rects_pos)

    ax_mm = fig.add_subplot(gs[0, 2])
    cats_mm = ['match', 'mismatch']; x3 = np.arange(len(cats_mm))
    rects_mm = ax_mm.bar(x3, mm_test)
    ax_mm.set_xticks(x3); ax_mm.set_xticklabels(cats_mm, rotation=45, ha='right')
    ax_mm.set_title('Match vs Mismatch (test)')
    annotate_bars(ax_mm, rects_mm)

    ax_comb = fig.add_subplot(gs[1, :])
    x4 = np.arange(len(all_comb_cats))
    rects_comb = ax_comb.bar(x4, comb_test_counts)
    ax_comb.set_xticks(x4); ax_comb.set_xticklabels(all_comb_cats, rotation=90, ha='right')
    ax_comb.set_title('Combined occlusion (test)')
    annotate_bars(ax_comb, rects_comb)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f'{DATASET}_test_occlusions_statistics.png'))
    plt.close(fig)



if __name__ == "__main__":
    outdir = ensure_output_dir()
    tv = load_trainval(OCCLUDED_TRAIN_VAL_SET_H5_PATH)
    test = load_test(OCCLUDED_TEST_SET_H5_PATH)

    class_names = tv['class_names']  # e.g. ['ANGRY', ...]
    # 1) match / mismatch (train vs val)
    cats_mm = ['match', 'mismatch']
    # in file mismatch field: 0=match, 1=mismatch
    mm_train = counts_for_categories(tv['mismatch_train'], [0,1])
    mm_val = counts_for_categories(tv['mismatch_val'], [0,1])
    plot_grouped_bar(cats_mm, mm_train, mm_val, 'train', 'val',
                     'Match vs Mismatch (train vs val)', os.path.join(outdir, f'{DATASET}_match_mismatch_trainval.png'))
    # test single
    mm_test = counts_for_categories(test['mismatch'], [0,1])
    plot_single_bar(cats_mm, mm_test, 'Match vs Mismatch (test)', os.path.join(outdir, f'{DATASET}_match_mismatch_test.png'))

    # 2) occlusion_type_emotion (train vs val)
    cats_em = class_names
    occ_train_labels = [class_names[i] for i in tv['occ_train']]
    occ_val_labels = [class_names[i] for i in tv['occ_val']]
    occ_train_counts = counts_for_categories(occ_train_labels, cats_em)
    occ_val_counts = counts_for_categories(occ_val_labels, cats_em)
    plot_grouped_bar(cats_em, occ_train_counts, occ_val_counts, 'train', 'val',
                     'Occlusion emotion type (train vs val)', os.path.join(outdir, f'{DATASET}_occ_emotion_trainval.png'))
    # test: occ_emotion from parsed strings; ensure same order (uppercase)
    test_occ_em_labels = [e.upper() for e in test['occ_emotion']]
    # If test contains emotions not in class_names, include them at end
    all_test_em = list(dict.fromkeys([e for e in test_occ_em_labels if e != 'UNKNOWN']))
    cats_em_test = class_names + [e for e in all_test_em if e not in class_names]
    occ_test_counts = counts_for_categories(test_occ_em_labels, cats_em_test)
    plot_single_bar(cats_em_test, occ_test_counts, 'Occlusion emotion type (test)', os.path.join(outdir, f'{DATASET}_occ_emotion_test.png'))

    # 3) occlusion_type_posneg (train vs val)
    cats_pos = ['negative', 'positive']
    pos_train_counts = counts_for_categories(tv['pos_train'], [0,1])
    pos_val_counts = counts_for_categories(tv['pos_val'], [0,1])
    plot_grouped_bar(cats_pos, pos_train_counts, pos_val_counts, 'train', 'val',
                     'Occlusion pos/neg (train vs val)', os.path.join(outdir, f'{DATASET}_occ_posneg_trainval.png'))
    # test
    pos_test_counts = counts_for_categories(test['posneg'], [0,1])
    plot_single_bar(cats_pos, pos_test_counts, 'Occlusion pos/neg (test)', os.path.join(outdir, f'{DATASET}_occ_posneg_test.png'))

    # 4) combined occlusion_type (emotion + posneg) -> e.g. 'ANGRY-positive'
    comb_cats = []
    for em in class_names:
        comb_cats.append(f"{em}-negative")
        comb_cats.append(f"{em}-positive")
    def make_combined_from_indices(occ_indices, pos_flags):
        labels = []
        for oi, pf in zip(occ_indices, pos_flags):
            em = class_names[int(oi)]
            pn = 'positive' if int(pf)==1 else 'negative'
            labels.append(f"{em}-{pn}")
        return labels
    comb_train_labels = make_combined_from_indices(tv['occ_train'], tv['pos_train'])
    comb_val_labels = make_combined_from_indices(tv['occ_val'], tv['pos_val'])
    comb_train_counts = counts_for_categories(comb_train_labels, comb_cats)
    comb_val_counts = counts_for_categories(comb_val_labels, comb_cats)
    plot_grouped_bar(comb_cats, comb_train_counts, comb_val_counts, 'train', 'val',
                     'Combined occlusion types (train vs val)', os.path.join(outdir, f'{DATASET}_occ_combined_trainval.png'))

    # test combined
    # test['occ_emotion'] contains emotion names, test['posneg'] 0/1
    comb_test_labels = [f"{e}-{('positive' if p==1 else 'negative')}" for e,p in zip(test_occ_em_labels, test['posneg'])]
    # ensure categories include any new from test
    all_comb_cats = comb_cats + [c for c in sorted(set(comb_test_labels)) if c not in comb_cats]
    comb_test_counts = counts_for_categories(comb_test_labels, all_comb_cats)
    plot_single_bar(all_comb_cats, comb_test_counts, 'Combined occlusion types (test)', os.path.join(outdir, f'{DATASET}_occ_combined_test.png'))

    plot_trainval_grid(outdir, class_names,
                       mm_train, mm_val,
                       occ_train_counts, occ_val_counts,
                       pos_train_counts, pos_val_counts,
                       comb_cats, comb_train_counts, comb_val_counts)

    plot_test_grid(outdir, cats_em_test, occ_test_counts,
                   pos_test_counts, mm_test, all_comb_cats, comb_test_counts)

    print("Grid plots written to:", outdir)
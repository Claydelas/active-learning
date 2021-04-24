# %%
import json
import matplotlib as m
m.use("pgf")
m.rcParams.update({
     'font.family': 'serif',
     'text.usetex': True,
     'pgf.rcfonts': False,
 })
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from scipy.signal import savgol_filter
# %%
def extract_vals(results):
    if results[0].get('name'):
        x, y = zip(*[(i['name'], i['macro avg']['f1-score']) for i in results])
    else:
        x, y = zip(*[(i['labels'], i['macro avg']['f1-score'])
                     for i in results])
    return x, y
# %%%%%
def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
# %%

def visualise(axes, title = None, ref = None, ref_name = None, trend=True):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fig, ax = plt.subplots(figsize=set_size(433.62))
    ax.grid(True)

    ax.set_xlabel("number of labels")
    ax.set_ylabel("macro f1-score")

    

    if ref:
        ax.axhline(y=ref[1], color='black', linestyle='-', alpha=0.8, label=ref_name)
    
    if title:
        fig.suptitle(title)

    for x, y, name in axes:
        if trend:
            y1 = savgol_filter(y, 101, 9)
            p = ax.plot(x, y1, linewidth=1, label=name)
            color = p[-1].get_color()
            ax.plot(x, y, alpha=0.2, linewidth=1, color=color)
        else:
            ax.plot(x[::2], y[::2], label=name)

    lg = ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', fancybox=True, shadow=True, ncol=2)
    fig.add_artist(lg)
    return fig, lg
# %%
def graph(path, title = None, ref = None, ref_name = None, trend = None):
    axes = []
    lim = []
    for file in glob.glob(path, recursive=False):
        x, y = extract_vals(json.load(open(file)))
        axes.append((x, y, os.path.splitext(os.path.basename(file))[0]))
    if ref:
        for r in glob.glob(ref, recursive=False):
            a, b = extract_vals(json.load(open(r)))
            lim.append((max(a),max(b)))
        ref = max(lim,key=lambda item:item[0])[0], max(lim,key=lambda item:item[1])[1]
    return visualise(axes, title, ref, ref_name, trend)
# %%
def save(files, path = '../example.pdf', type = 'pgf', title = None, ref=None, ref_name = None, trend = None):
    fig, extra = graph(files, title, ref, ref_name, trend)
    
    if type == 'pdf':
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(path)
        pp.savefig(fig, dpi=1000)
        pp.close()
    else:
        fig.savefig(path, 
                     dpi=1000, 
                     format=type,
                     bbox_extra_artists=(extra,),
                     bbox_inches='tight')
    return fig, extra
# %%
save("../vis/KNN/*.json",
    path='../graphs/KNN-all.pdf',
    type='pdf',
    title="Active Learning with K-Nearest Neighbors",
    ref="../vis/KNN/ML/*.json",
    ref_name="Max KNN Baseline ML Score",
    trend=True)
# %%
save("../vis/LR/*.json",
    path='../graphs/LR-all.pdf',
    type='pdf',
    title="Active Learning with Logistic Regression",
    ref="../vis/LR/ML/*.json",
    ref_name="Max LR Baseline ML Score")
# %%
save("../vis/RF/*.json",
    path='../graphs/RF-all.pdf',
    type='pdf',
    title="Active Learning with Random Forest",
    ref="../vis/RF/ML/*.json",
    ref_name="Max RF Baseline ML Score")
# %%
save("../vis/LSVM/*.json",
    path='../graphs/LSVM-all.pdf',
    type='pdf',
    title="Active Learning with Linear SVM",
    ref="../vis/LSVM/ML/*.json",
    ref_name="Max LSVM Baseline ML Score")
# %% LR Text+Stats TDAVIDSON
graph("../vis/External/LR/Text-Stats/*.json",
    title="Logistic Regression with Text and Statistical Features",
    ref="../vis/External/LR/Text-Stats/ML/*.json",
    ref_name="Logistic Regression Target Score (21297 labels)")
# %% LR Text TDAVIDSON
graph("../vis/External/LR/Text/*.json",
    title="Logistic Regression with Text Features",
    ref="../vis/External/LR/Text/ML/*.json",
    ref_name="Logistic Regression Target Score (21297 labels)")
# %% LSVM Text+Stats TDAVIDSON
graph("../vis/External/LSVM/Text-Stats/*.json",
    title="Linear SVM with Text and Statistical Features",
    ref="../vis/External/LSVM/Text-Stats/ML/*.json",
    ref_name="Linear SVM Target Score (21297 labels)")
# %% LSVM Text TDAVIDSON
graph("../vis/External/LSVM/Text/*.json",
    title="Linear SVM with Text Features",
    ref="../vis/External/LSVM/Text/ML/*.json",
    ref_name="Logistic Regression Target Score (21297 labels)")

#%% AL extract







# %% Text TDAVIDSON 26K
save("../vis/External Results on 26k/Text/*.json",
    path='../graphs/t-davidson-query-strat-differential.pgf',
    ref="../vis/External Results on 26k/Text/ML/*.json",
    ref_name="Baseline Score (21297 labels)")

# %% Text + Stats TDAVIDSON 26K
save("../vis/External Results on 26k/Text + Stats/*.json",
    path='../graphs/t-davidson-query-strat-differential+stats.pgf',
    ref="../vis/External Results on 26k/Text + Stats/ML/*.json",
    ref_name="Baseline Score (21297 labels)")
# %% personal LR TFIDF
save("../vis/new personal/LR-TFIDF/*.json",
    path='../graphs/personal-lr-tfidf.pgf',
    ref="../vis/new personal/LR-TFIDF/ML/*.json",
    ref_name="Maximum Baseline Score (1720 labels)",
    trend=False)
# %% personal RF d2v
save("../vis/new personal/RF-D2V/*.json",
    path='../graphs/personal-rf-d2v.pgf',
    ref="../vis/new personal/RF-D2V/ML/*.json",
    ref_name="Maximum Baseline Score (1720 labels)",
    trend=False)
# %%
# %% personal LSVM GLOVE
save("../vis/new personal/LSVM-GLOVE/*.json",
    path='../graphs/personal-lsvm-glove.pgf',
    ref="../vis/new personal/LSVM-GLOVE/ML/*.json",
    ref_name="Maximum Baseline Score (1720 labels)",
    trend=False)
# %%
graph("../vis/RF/*GloVe*Uncertainty Sampling-AL*.json",
    ref="../vis/RF/ML/*GloVe*.json",
    ref_name="Maximum Baseline Score (1720 labels)",
    trend=False)
# %%
graph("../vis/RF/*TFIDF*Uncertainty Sampling-AL*.json",
    ref="../vis/RF/ML/*TFIDF*.json",
    ref_name="Maximum Baseline Score (1720 labels)",
    trend=False)
# %%
graph("../vis/KNN/*Doc2Vec*Uncertainty Sampling-AL*.json",
    ref="../vis/KNN/ML/*Doc2Vec*.json",
    ref_name="Maximum Baseline Score (1720 labels)",
    trend=False)
# %%
graph("../vis/External Results on 26k/Text/*.json",
    ref="../vis/External Results on 26k/Text/ML/*.json",
    ref_name="Baseline Score (21297 labels)")
# %%
def al_vis(al_path, baseline_path):
    print("--------------------------------------------")
    print(f'{os.path.abspath(al_path)}')
    xb, yb = extract_vals(json.load(open(baseline_path)))
    xb, yb = np.array(xb), np.array(yb)
    print(f'ML baseline score @{xb} with {round(yb[0] * 100, 2)} macro F1. ({yb[0]})')
    x, y = extract_vals(json.load(open(al_path)))
    x, y = np.array(x), np.array(y)
    max_idx = np.argmax(y)
    al_gt = np.argwhere(y >= yb)
    least_labels = al_gt[0] if al_gt.size else np.nan
    al_gt2 = np.argwhere(y >= yb-0.02)
    least_labels2 = al_gt2[0] if al_gt2.size else np.nan
    print(f'max @{x[max_idx]} with [{round(y[max_idx] * 100, 2)}] macro F1. ({y[max_idx]})')
    print(f'reached baseline @{x[least_labels]}')
    print(f'2% baseline margin @{x[least_labels2]} with [{round(y[least_labels2][0] * 100, 2)}] macro F1. ({y[least_labels2][0]})')
    print(f'eff gain [{round((((xb[0] - x[least_labels][0]) / xb[0]) * 100), 2)}\%]')
# %%
al_vis("D:/dev/Mixed/active-learning/vis/new personal\LR-TFIDF\Text Features.json",
    "D:/dev/Mixed/active-learning/vis/new personal\LR-TFIDF\ML\personal-LR-TFIDF-T-ML.json")
# %%
al_vis("D:/dev/Mixed/active-learning/vis/new personal\LR-TFIDF\Text + User Features.json",
    "D:/dev/Mixed/active-learning/vis/new personal\LR-TFIDF\ML\personal-LR-TFIDF-T-U-ML.json")
# %%
al_vis("D:/dev/Mixed/active-learning/vis/new personal\LR-TFIDF\Text + Statistical Features.json",
    "D:/dev/Mixed/active-learning/vis/new personal\LR-TFIDF\ML\personal-LR-TFIDF-T-S-ML.json")
# %%
al_vis("D:/dev/Mixed/active-learning/vis/new personal\LR-TFIDF\Text + User + Statistical Features.json",
    "D:/dev/Mixed/active-learning/vis/new personal\LR-TFIDF\ML\personal-LR-TFIDF-T-U-S-ML.json")

# %%
def al_results(als, mls):
    for al in glob.glob(als, recursive=False):
        al_name = os.path.splitext(os.path.basename(al))[0].split('-')[:-1]
        if al_name[-1] == 'Uncertainty Sampling':
            al_name = "-".join(al_name[:-1])
            for ml in glob.glob(mls, recursive=False):
                ml_name = "-".join(os.path.splitext(os.path.basename(ml))[0].split('-')[:-1])
                if al_name == ml_name:
                    al_vis(al, ml)

# %% KNN
al_results("D:/dev/Mixed/active-learning/vis/KNN/*.json", "D:/dev/Mixed/active-learning/vis/KNN/ML/*.json")
# %% LSVM
al_results("D:/dev/Mixed/active-learning/vis/LSVM/*.json", "D:/dev/Mixed/active-learning/vis/LSVM/ML/*.json")
# %% RF
al_results("D:/dev/Mixed/active-learning/vis/RF/*.json", "D:/dev/Mixed/active-learning/vis/RF/ML/*.json")
# %% LR
al_results("D:/dev/Mixed/active-learning/vis/LR/*.json", "D:/dev/Mixed/active-learning/vis/LR/ML/*.json")
# %%
from wordcloud import WordCloud
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
personal = pd.read_pickle("../data/processed/personal_processed.pkl")
c1 = CountVectorizer(max_features=10000,stop_words='english')
bad_tweets = c1.fit_transform(personal[personal.target == 1].tweet)
bad_word_list = c1.get_feature_names()
bad_count_list = np.asarray(bad_tweets.sum(axis=0))[0]

c2 = CountVectorizer(max_features=10000,stop_words='english')
good_tweets = c2.fit_transform(personal[personal.target == 0].tweet)
good_word_list = c2.get_feature_names()
good_count_list = np.asarray(good_tweets.sum(axis=0))[0]

freq_bad = dict(zip(bad_word_list, bad_count_list))
freq_good = dict(zip(good_word_list, good_count_list))

wordcloud = WordCloud(background_color="white")
wordcloud.generate_from_frequencies(frequencies=freq_bad)
wordcloud2 = WordCloud(background_color="white")
wordcloud2.generate_from_frequencies(frequencies=freq_good)

wordcloud.to_file("../results/bad2.png")
wordcloud2.to_file("../results/good2.png")
# %%
tdavidson = pd.read_csv("../data/t-davidson/labeled_data.csv")
# %%
plt.rc('font', family='serif')
plt.rc('ytick', labelsize='x-small')
plt.rc('xtick', labelsize='small')
fig, ax = plt.subplots(figsize=set_size(200))
ax.xaxis.set_ticks_position('none') 
ax.set_ylabel("tweets")
dist = tdavidson['class'].value_counts()
ax.bar(['hate speech','offensive','neither'], [dist[0],dist[1],dist[2]])
fig.savefig("../results/tdavidson-class-balance.pgf", 
                     dpi=1000, 
                     format="pgf",
                     bbox_inches='tight', pad_inches = 0)
# %%
al_results("D:/dev/Mixed/active-learning/backend/test/results/External Results on 26k/Text/*AL*.json",
"D:/dev/Mixed/active-learning/backend/test/results/External Results on 26k/Text/*ML*.json")
# %%
al_results("D:/dev/Mixed/active-learning/backend/test/results/External Results on 26k/Text + Stats/*AL*.json",
"D:/dev/Mixed/active-learning/backend/test/results/External Results on 26k/Text + Stats/*ML*.json")
# %%

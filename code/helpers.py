import json
import pickle
import os

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


# instantiate wordnet lemmatizer
wn = WordNetLemmatizer()
# instantiate Regex tokenizer
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
# define english stopwords
english_stopwords = set(stopwords.words("english"))


mapper = {
    'J': wordnet.ADJ,
    'V': wordnet.VERB,
    'N': wordnet.NOUN,
    'R': wordnet.ADV
}


def write_to_json(scores, filename='scores'):

    if f'{filename}.json' in os.listdir('../data/'):
        with open(f'../data/{filename}.json') as f:
            prev_scores = json.load(f)
            scores = prev_scores | scores

    with open(f'../data/{filename}.json', 'w') as f:
        json.dump(scores, f)


def read_json_file(filename):

    try:
        with open(f"../data/{filename}.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"{filename} is not a valid file name.")


def get_scores_dict(key, train_acc, test_acc):
    return {
        key: {"Train accuracy": train_acc, "Test accuracy": test_acc}
    }


def separate_X_y(df, features, target):
    return df[features], df[target]


def custom_lemmatize(word_tag):
    word, tag = word_tag
    pos = mapper.get(tag[0])
    return wn.lemmatize(word, pos) if pos else word


def lemmatized(sentence):
    tokens = tokenizer.tokenize(sentence.lower())
    tokens_no_stopwords = [w for w in tokens if w not in english_stopwords]
    return ' '.join(map(custom_lemmatize, pos_tag(tokens_no_stopwords)))



def save_model(filename, model):
    with open(f"../models/{filename}.model", "wb") as f:
        pickle.dump(model, f)
        
def load_model(model_name):
    
    try:
        with open(f"../models/{model_name}.model", 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise ValueError(f"{filename} is not a valid model.")


def plot_bars(data, title, x, y, hue=None, size=(12,5), baseline=None, to_filename=None, fontsizes=None):

    if fontsizes is None:
        fontsizes = {'title': 20, 'label': 12}
        
    f, ax = plt.subplots(1, figsize=size)
    
    # plot bars, set title and labels
    sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax)
    ax.set_title(title, fontsize=fontsizes['title'], pad=20)
    ax.xaxis.set_tick_params(labelsize=fontsizes['label'])
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.set_xlabel('')
    sns.despine(f)
    
    # plot baseline value
    if baseline:
        ax.axhline(baseline, label='Baseline', color='red', linestyle='-.', alpha=0.7)
        ax.set_ylim(round(baseline - 0.1, 1), 1)
    
    # label bars
    for bar in ax.patches:
        bar_height = round(bar.get_height(), 3)
        ax.annotate(bar_height, (bar.get_x() + bar.get_width()/2, bar.get_height()*1.01), 
                    ha='center', color='black', fontsize=10)
    
    # set legend
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=13)
    f.tight_layout();
    
    # save to file
    if to_filename:
        f.savefig(f'../images/{to_filename}.png', transparent=True, bbox_inches="tight")



def plot_most_common(X, name='', size=10, filename=None, **kwargs):

    cvec = CountVectorizer(**kwargs)
    X_ = cvec.fit_transform(X)

    # index of the most common 10 bigrams
    arr = np.array(X_.sum(axis=0)).ravel()
    idx_most_common = np.argpartition(arr, -size)[-size:]
    # get bigrams
    features_names = cvec.get_feature_names_out()
    # most common 10 counts
    vals = arr[idx_most_common]
    # index of sorted vals
    ix = np.argsort(vals)

    # plot bars
    f, ax = plt.subplots(1, figsize=(12, 0.7*size))
    sns.barplot(x=vals[ix], y=features_names[idx_most_common][ix], ax=ax)
    ax.set_title(f"{size} most common words {'in ' + name}", fontsize=20)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.set_xlabel("Count", fontsize=15)

    # label bars
    ax.bar_label(ax.containers[0], label_type='edge')
#    ax.margins(y=0.1)
    sns.despine(f)
    f.tight_layout();

    if filename:
        f.savefig(f"../images/{filename}.png", bbox_inches="tight", transparent=True);
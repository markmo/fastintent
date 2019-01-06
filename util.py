import numpy as np
import pandas as pd
from pathlib import Path
import re


# noinspection SpellCheckingInspection
def clean_text(text):
    # remove non-latin chars (`np.savetxt` supports only latin-1)
    text = re.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', '', text)
    # remove whitespace at ends
    text = text.strip()
    # replace whitespace with single spaces
    text = re.sub(r'\s+', ' ', text)
    # remove non-printable chars
    text = ''.join(ch for ch in text if ch.isprintable())
    return text


def convert_label(label):
    """ Conform to expected fasttext label format. """
    return '__label__{}'.format(label)


def preprocess_csv():
    """ Convert exported CSV file from Watson to fasttext format. """
    root_path = Path(__file__).parent
    intents_path = str(root_path / 'models/intents.csv')
    df = pd.read_csv(intents_path, header=None)
    df = df.dropna()
    df[0] = df[0].apply(clean_text)
    df[1] = df[1].apply(convert_label)

    # save to file
    out_path = str(root_path / 'models/intents.txt')
    np.savetxt(out_path, df.values, fmt='%s')

    return df

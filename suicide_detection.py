import spacy
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
from wordcloud import WordCloud
from collections import Counter
import re
import utils

nlp = spacy.load("en_core_web_sm")


def gen_wordcloud(texts: pd.Series):
    joined = " ".join(texts)
    bow = Counter(re.findall("\w+", joined))
    wc = WordCloud(
        background_color="white", contour_color="steelblue", contour_width=2
    ).generate_from_frequencies(frequencies=dict(bow))
    return wc


def generate_lemmatized_tokens(text: str):
    valid_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    tokens = nlp(text)
    for token in tokens:
        if not token.is_alpha or token.is_stop:
            continue
        elif token.pos_ in valid_pos:
            yield token.lemma_
        else:
            yield token.text


def lemmatize(text: str):
    valid_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    tokens = nlp(text)
    for token in tokens:
        if not token.is_alpha or token.is_stop:
            continue
        elif token.pos_ in valid_pos:
            yield token.lemma_
        else:
            yield token.text


def tokenize(text: str):
    return " ".join(list(generate_lemmatized_tokens(text)))


if __name__ == "__main__":
    # CONFIG
    N_DOCS = 50000
    FILE_PATH = "Suicide_Detection.csv"
    DOWNLOAD_DATASET = False
    LOAD_MODEL = False
    SAVE_MODEL = True

    url = "https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/download?datasetVersionNumber=14"
    model_filename = "gnb_fit.sav"

    if DOWNLOAD_DATASET:
        utils.download_dataset(url)

    df = pd.read_csv(FILE_PATH, delimiter=",", header=0, index_col=0)
    if N_DOCS is not None:
        df = df[:N_DOCS]
    tqdm.pandas(desc="LEMMATIZED")
    df["text"] = df["text"].progress_apply(tokenize)

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["class"], test_size=0.2, random_state=1999
    )
    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)

    x_train_vect = vectorizer.transform(x_train)
    x_test_vect = vectorizer.transform(x_test)

    gnb = MultinomialNB()

    if LOAD_MODEL:
        model = pickle.load(open(model_filename, "rb"))
    else:
        model = gnb.fit(x_train_vect.toarray(), y_train)
    if SAVE_MODEL:
        pickle.dump(model, open(model_filename, "wb"))

    prediction = model.predict(x_test_vect.toarray())

    # Wordclouds
    wc_suicide = gen_wordcloud(df[df["class"] == "suicide"]["text"])
    wc_non_suicide = gen_wordcloud(df[df["class"] == "non-suicide"]["text"])
    plt.imshow(wc_suicide, interpolation="bilinear")
    plt.title("Suicide wordcloud")
    plt.axis("off")
    plt.show()
    plt.imshow(wc_non_suicide, interpolation="bilinear")
    plt.title("Non-Suicide wordcloud")
    plt.axis("off")
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_test, prediction, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()

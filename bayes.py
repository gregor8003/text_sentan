import itertools
import sys

from sklearn import metrics
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    HashingVectorizer,
    TfidfTransformer,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    get_label_from_rating,
    get_mdsd_csv_file,
    POSITIVE,
    NEGATIVE,
    UNLABELED,
    FIELD_LABEL,
    FIELD_RATING,
    FIELD_PREPROCESSED,
)
from utils_plot import plot_confusion_matrix, plot_learning_curve


LABELS = {
    'values': (POSITIVE, NEGATIVE),
    'names': ('Positive', 'Negative'),
}

MAIN_PATH = 'mdsd'

PLOT_TRAIN_CM = True

PLOT_VERIFICATION_CM = True

PLOT_LEARNING_CURVE = True

#####

TFIDF_PIPE = (
    (
        'tfidf',
        TfidfVectorizer,
        {
            'strip_accents': None,
            'lowercase': True,
            'preprocessor': None,
            'stop_words': None,
            'analyzer': str.split,
        },
    ),
)

HASHING_TFIDF_PIPE = (
    (
        'hashing',
        HashingVectorizer,
        {
            'non_negative': True,
            ###
            'strip_accents': None,
            'lowercase': True,
            'preprocessor': None,
            'stop_words': None,
            'analyzer': str.split,
        },
    ),
    (
        'tfidf',
        TfidfTransformer,
        {
        },
    ),
)


def custom_pipeline(steps=[]):
    step_insts = []
    for step_name, step_klass, step_opts in steps:
        step_inst = step_klass(**step_opts)
        if step_name is None:
            step_name = step_klass.__name__.tolower()
        step_insts.append((step_name, step_inst))
    pipeline = Pipeline(steps=step_insts)
    return pipeline


def shuffle_X_y(X, y):
    assert X.shape[0] == y.shape[0]
    p = np.random.permutation(X.shape[0])
    return X[p], y[p]


def reshape_labels(labels):
    return np.array(labels).reshape(len(labels), 1)


def main():

    main_path = sys.argv[1]

    mdsd_csv_file = get_mdsd_csv_file(main_path=main_path)

    df = pd.read_csv(mdsd_csv_file, encoding='utf-8')

    # print empty documents if present - not suitable for NB
    # print(df[pd.isnull(df[FIELD_TEXT])])

    idxs = {}
    for label in (POSITIVE, NEGATIVE, UNLABELED):
        idxs[label] = df.index[df[FIELD_LABEL] == label]

    posneg_idxs = list(itertools.chain(idxs[POSITIVE], idxs[NEGATIVE]))

    documents = df.loc[posneg_idxs, FIELD_PREPROCESSED]
    labels = df.loc[posneg_idxs, FIELD_LABEL]
    labels = reshape_labels(labels)

    feature_vectors = custom_pipeline(HASHING_TFIDF_PIPE)

    print('Transforming data', '... ', end='', flush=True)
    X_data = feature_vectors.fit_transform(documents)
    print('done')

    X_idxs_train, X_idxs_test, y_idxs_train, y_idxs_test = (
        train_test_split(
            range(X_data.shape[0]), labels, test_size=0.3,
            random_state=0, stratify=labels
        )
    )

    X_train = X_data[X_idxs_train]
    X_test = X_data[X_idxs_test]
    y_train = y_idxs_train
    y_test = y_idxs_test

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    X_train, y_train = shuffle_X_y(X_train, y_train)

    print('Fitting', '... ', end='', flush=True)
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train.ravel())
    print('done')

    print('Predicting on test set', '... ', end='', flush=True)
    y_pred = classifier.predict(X_test)
    print('done')

    report = metrics.classification_report(
        y_test, y_pred,
        labels=LABELS['values'], target_names=LABELS['names'], digits=6
    )
    print(report)

    if PLOT_TRAIN_CM:
        plt.figure()
        cm = metrics.confusion_matrix(y_test, y_pred, LABELS['values'])
        plot_confusion_matrix(plt, cm, LABELS['names'], normalize=False)
        plt.show()

    unl_idxs = idxs[UNLABELED]
    unl_documents = df.loc[unl_idxs, FIELD_PREPROCESSED]
    # remove NaNs (empty strings)
    unl_documents = unl_documents[pd.notnull(unl_documents)]

    unl_ratings = df.loc[unl_documents.index, FIELD_RATING]

    y_rating = [get_label_from_rating(r) for r in unl_ratings]
    y_rating = reshape_labels(y_rating)

    print('Transforming data', '... ', end='', flush=True)
    X_test_unl = feature_vectors.transform(unl_documents)
    print('done')

    print(X_test_unl.shape)

    print('Predicting on validation set', '... ', end='', flush=True)
    y_pred = classifier.predict(X_test_unl)
    print('done')

    report = metrics.classification_report(
        y_rating, y_pred,
        labels=LABELS['values'], target_names=LABELS['names'], digits=6
    )
    print(report)

    if PLOT_VERIFICATION_CM:
        plt.figure()
        cm = metrics.confusion_matrix(y_rating, y_pred, LABELS['values'])
        plot_confusion_matrix(plt, cm, LABELS['names'], normalize=False)
        plt.show()

    if PLOT_LEARNING_CURVE:
        X_all, y_all = shuffle_X_y(X_data, labels)
        print('Plotting learning curve', '... ', end='', flush=True)
        plot_learning_curve(
            plt, classifier, 'Learning curve', X_all, y_all.ravel(),
            cv=10, n_jobs=1
        )
        print('done')
        plt.show()


if __name__ == '__main__':
    main()

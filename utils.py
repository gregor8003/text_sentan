import os
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk


MDSD_MAIN_PATH = 'mdsd'

MDSD_DOMAIN_PATHS = (
    'books',
    'dvd',
    'electronics',
    'kitchen_&_housewares',
)

MDSD_DOMAIN_FILES = (
    'negative.review',
    'positive.review',
    'unlabeled.review'
)

MDSD_CSV_FILE = 'mdsd.csv'


def _get_domain_paths(main_path, domain_paths, domain_files, exclusions=()):
    paths = []
    for domain_path in domain_paths:
        for domain_file in domain_files:
            path = os.path.join(main_path, domain_path, domain_file)
            # get chars until first .
            domain_file_part = domain_file[:domain_file.find('.')]
            if path not in exclusions:
                paths.append((path, domain_path, domain_file_part))
    return paths


def get_mdsd_domain_paths(main_path=MDSD_MAIN_PATH,
                          domain_paths=MDSD_DOMAIN_PATHS,
                          domain_files=MDSD_DOMAIN_FILES,
                          exclusions=()):
    return _get_domain_paths(main_path, domain_paths, domain_files, exclusions)


def _get_file_path(main_path, file_path):
    return os.path.join(main_path, file_path)


def get_mdsd_csv_file(main_path=MDSD_MAIN_PATH, file_path=MDSD_CSV_FILE):
    return _get_file_path(main_path, file_path)


PARTIALS = {
    '\'s': (),
    '\'m': ('am',),
    'n\'t': ('not',),
    '\'ve': ('have',),
    '\'ll': ('will'),
    'can\'t': ('can', 'not',),
    'cannot': ('can', 'not',),
    'should\'ve': ('should', 'have',),
}


def process_partials(word):
    return PARTIALS.get(word, (word,))


STOPWORDS = set(stopwords.words('english'))

STEMMER = PorterStemmer()


def process_raw_word(word):
    return STEMMER.stem(word)


# !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
# no '
PUNCTUATION = r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~""" + string.digits

PUNCT_TRANSTABLE = str.maketrans('', '', PUNCTUATION)


def remove_punctuation(word, trans_table=PUNCT_TRANSTABLE):
    return word.translate(trans_table)


class SentenceTokenizer(object):

    def process(self, document):
        for sentence in nltk.sent_tokenize(document):
            for word in nltk.word_tokenize(sentence):
                for p in process_partials(word):
                    w = process_raw_word(remove_punctuation(p))
                    ws = frozenset((w,))
                    if not ws & STOPWORDS:
                        if len(w) > 0 and w != "''":
                            yield w

    def __call__(self, document):
        yield from self.process(document)


UNKNOWN = 0
POSITIVE = 1
NEGATIVE = 2
UNLABELED = 3

FIELD_DOMAIN = 'Domain'
FIELD_TEXT = 'Original Text'
FIELD_PREPROCESSED = 'Preprocessed Text'
FIELD_RATING = 'Rating'
FIELD_LABEL = 'Label'

CSV_FIELD_NAMES = [
    FIELD_TEXT, FIELD_PREPROCESSED, FIELD_DOMAIN, FIELD_RATING, FIELD_LABEL
]


def get_label(domain):
    if domain.startswith('positive'):
        return POSITIVE
    elif domain.startswith('negative'):
        return NEGATIVE
    elif domain.startswith('unlabeled'):
        return UNLABELED
    return UNKNOWN


def get_label_from_rating(rating):
    if rating in (4, 5):
        return POSITIVE
    elif rating in (1, 2):
        return NEGATIVE
    else:
        return None

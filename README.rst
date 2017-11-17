text_sentan
===========

This repository contains examples of Sentiment Analysis, a popular topic of
text processing.

Installation
------------

- Install latest `Anaconda3 <https://www.anaconda.com/download>`_

- Download and unpack latest text_sentan source distribution, or simply clone the
  repo

- Create conda environment

.. code-block:: bash

    $ cd text_sentan-<x.y.z.www>
    $ conda env create -f environment.yml
    $ activate text_sentan

- See `Dataset(s)`_ for preparing data

- See `Usage`_ for scripts

Dataset(s)
----------

NOTE: dataset(s) are not included and must be downloaded separately.

MDSD
^^^^

Multi-Domain Sentiment
Dataset (MDSD) version 1 (https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)
contains Amazon reviews from four categories: books, dvd, electronics,
kitchen & housewares. They are rated from 1 to 5, where 1 or 2 means "negative",
and 4 or 5 means "positive". The dataset contains labeled and unlabeled data.

* Download https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz and unpack to the directory of choice.

* Move/copy four subdirectories with categories data to directory named ``mdsd``, or simply rename unpacked one.

* Download https://www.cs.jhu.edu/~mdredze/datasets/sentiment/book.unlabeled.gz, unpack it and place as ``mdsd/books/unlabeled.review``.

You should end up with the following directory structure:

.. code-block:: bash

    +---mdsd
        |
        +---books
        |     negative.review
        |     positive.review
        |     unlabeled.review
        +---dvd
        |     negative.review
        |     positive.review
        |     unlabeled.review
        +---electronics
        |     negative.review
        |     positive.review
        |     unlabeled.review
        +---kitchen_&_housewares
              negative.review
              positive.review
              unlabeled.review

Usage
-----

mdsd_to_csv
^^^^^^^^^^^

.. code-block:: bash

    $ python3 -m mdsd_to_csv [path_to_mdsd_dir]

The ``mdsd_to_csv`` script reads data files from ``mdsd`` directory (with
structure described above), and produces CSV file ``mdsd.csv`` in ``mdsd``
directory.

After the script finishes successfully, you should end up with:

.. code-block:: bash

    +---mdsd
        |  mdsd.csv
        +---books
        ..

bayes
^^^^^

.. code-block:: bash

    $ python3 -m bayes [path_to_mdsd_dir]

The ``bayes`` script reads ``ndsd.csv`` file from ``mdsd`` directory, and
performs sentiment analysis via Naive Bayes classifier. It produces classification
reports on both test part and validation part, together with corresponding plots
of confusion matrices. In addition, it plots learning curve.


References
----------

Blitzer J., Dredze M., Pereira F. "Biographies, Bollywood, Boom-boxes and 
Blenders: Domain Adaptation for Sentiment Classification.", Association of
Computational Linguistics (ACL), 2007

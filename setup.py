from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='text_sentan',
    version='0.1.0',
    description='Sentiment analysis examples',
    long_description=long_description,
    author='Grzegorz Zycinski',
    author_email='g.zycinski@gmail.com',
    url='http://github.com/gregor8003/text_sentan/',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Text Processing :: General',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='text processing sentiment analysis',
    py_modules=['mdsd_to_csv', 'bayes', 'utils', 'utils_plot'],
    zip_safe=False
)

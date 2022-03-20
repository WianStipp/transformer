#!/bin/bash

# Downloads the English-German dataset from Stanford's NMT group: https://nlp.stanford.edu/projects/nmt/
# WMT'14 English-German data [Medium]

wget -P data/en-de https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en
wget -P data/en-de https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de
wget -P data/en-de https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.en
wget -P data/en-de https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.de
wget -P data/en-de https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.en
wget -P data/en-de https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.de
wget -P data/en-de https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/dict.en-de

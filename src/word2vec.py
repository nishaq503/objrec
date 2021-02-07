import os
import numpy as np
from typing import List
import click

import fasttext
import csv

FILE = 'foobar.csv'

SRC = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(SRC, '..'))


CLAMP = False


def loader(file) -> List[str]:
    """Load a data file and return it as a sentence."""
    with open(os.path.join(ROOT, f'{file}'), 'r') as fp:
        sentences = list(csv.reader(fp))
        if CLAMP:
            sentence = [','.join([str(int(bool(int(c)))) for c in w]) for w in sentences]
        else:
            sentence = [','.join(w) for w in sentences]
    return sentence


@click.group()
def cli():
    pass


@cli.command()
@click.argument('files', nargs=-1)
@click.option('--output', default='digested.txt')
def digest(files, output):
    """Digest the provided files into a corpus written to output."""
    # TODO: Tom, please make this feel nicer
    folder = 'digested'
    os.makedirs(os.path.join(ROOT, folder), exist_ok=True)
    filepath = os.path.join(ROOT, folder, output)
    if os.path.exists(filepath):
        os.remove(filepath)

    sentences = (loader(f) for f in files)
    for i, s in enumerate(sentences):
        print(f'sentence {i} with {len(s)} words')
        with open(filepath, 'a') as fp:
            fp.write(' '.join([w for w in s]))
            fp.write('\n')


def _digest(file, output):
    with open(os.path.join(ROOT, output), 'w') as fp:
        fp.write(' '.join([w for w in loader(file)]))
        fp.write('\n')


def _ingest(model, target):
    with open(os.path.join(ROOT, target), 'r') as fp:
        target_data = fp.readlines()
    
    for i, sentence in enumerate(target_data):
        # noinspection PyBroadException
        try:
            np.save(f'{target}.{i}.npy', np.stack([model[word] for word in sentence.split(' ')]))
        except Exception as _:
            print('fucking fucked it')


def _train(corpus):
    return fasttext.train_unsupervised(corpus, model='skipgram')


@cli.command()
@click.argument('corpus')
@click.argument('target')
def ingest(corpus, target):
    """Ingest a corpus file and produce numpy memmaps."""
    model = _train(corpus)
    _ingest(model, target)


@cli.command()
@click.argument('corpus')
@click.argument('files', nargs=-1)
def splay(corpus, files):
    """Splay all of the given files into individual numpy memmaps. The plural version of ingest."""
    model = _train(corpus)
    for file in files:
        _digest(file, f'{file}.digested') 
        _ingest(model, f'{file}.digested')


if __name__ == "__main__":
    cli()

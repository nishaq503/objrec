import os
from typing import List
import click

from gensim.models import Word2Vec
import csv

FILE = 'foobar.csv'

SRC = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(SRC, '..'))

def loader(file) -> List[str]:
    """Load a data file and return it as a sentence."""
    with open(os.path.join(ROOT, f'{file}'), 'r') as fp:
        sentences = list(csv.reader(fp))
        sentence = [''.join([str(int(bool(int(c)))) for c in w]) for w in sentences]
    return sentence
    
@click.group()
def cli():
    pass

@cli.command()
@click.argument('files', nargs=-1)
def digest(files):
    sentences = (loader(f) for f in files)
    with open(os.path.join(ROOT, 'digested.txt'), 'w') as fp:
        fp.write('\n'.join((' '.join([w for w in s]) for s in sentences)))

@cli.command()
@click.argument('corpus')
def ingest(corpus):
    with open(corpus, 'r') as fp:
        corpus = fp.readlines()
    model = Word2Vec(sentences=[s.split(' ') for s in corpus])
    breakpoint()

@cli.command()
@click.argument('files', nargs=-1)
def predict(files):
    breakpoint()
    model = Word2Vec(sentences=[loader(f) for f in files])

if __name__ == "__main__":
    cli()

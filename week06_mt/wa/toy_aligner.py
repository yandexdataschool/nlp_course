import os, sys, codecs

from utils import read_parallel_corpus
from utils import write_aligned_corpus
from utils import extract_test_sets

def count_word_cooccurrences(src_corpus, trg_corpus):
    # Counts how often pairs of words co-occur in a sentence pair.
    counts = {}
    for i, src_sent in enumerate(src_corpus):
        for src in src_sent:
            if not src in counts:
                counts[src] = {}
            for trg in trg_corpus[i]:
                if not trg in counts[src]:
                    counts[src][trg] = 0
                counts[src][trg] += 1
    return counts

def align_corpus(src_corpus, trg_corpus, counts):
    # Align each target word with most commonly seen source word.
    aligned_corpus = []
    for src_sent, trg_sent in zip(src_corpus, trg_corpus):
        alignments = {}
        for j, trg in enumerate(trg_sent):
            if trg not in counts:
                continue
            max_count, best_src = 0, -1
            for i, src in enumerate(src_sent):
                if src not in counts[trg]:
                    continue
                if counts[src][trg] > max_count:
                    best_src = i 
                    max_count = counts[src][trg]
            if best_src > -1:
                alignments[j] = { best_src : '*' }
        aligned_corpus.append((src_sent, trg_sent, alignments))
    return aligned_corpus

if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("Usage ./toy_aligner.py parallel-corpus output-suffix")
        sys.exit(0)
    parallel_text, output_suffix = sys.argv[1:3]
    src_corpus, trg_corpus, _ = read_parallel_corpus(parallel_text)
    print('read corpus of %d parallel sentences.' % (len(src_corpus)))
    counts = count_word_cooccurrences(src_corpus, trg_corpus)
    aligned_corpus = align_corpus(src_corpus, trg_corpus, counts)
    print('aligned corpus of %d parallel sentences.' % (len(aligned_corpus)))
    # write out test set alignments only
    test_sets = extract_test_sets(aligned_corpus)
    print(test_sets)
    for name, aligned_sentences in test_sets.items():
        write_aligned_corpus(aligned_sentences, '%s-%s' % (name, output_suffix))
    

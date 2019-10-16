#!/usr/bin/python
import os, sys, codecs

def read_all_tokens(path):
    return [line.strip().split() for line in codecs.open(path, 'r', 'utf8')]

def count_word_cooccurrences(src_corpus, trg_corpus):
    # Counts how many times each pair of source and target words occur in
    # the same sentence pair.
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
    # Aligns each source word with the most commonly associated target word
    alignments = []
    for i, src_sent in enumerate(src_corpus):
        alignment = {}
        for j, src in enumerate(src_sent):
            if src not in counts:
                continue
            max_count, best_trg = 0, -1
            for k, trg in enumerate(trg_corpus[i]):
                if trg not in counts[src]:
                    continue
                if counts[src][trg] > max_count:
                    best_trg = k
                    max_count = counts[src][trg]
            if best_trg > -1:
                alignment[j] = best_trg
        alignments.append(alignment)
    return alignments

if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("Usage ./toy_aligner.py src_corpus trg_corpus > wa_output.")
        sys.exit(0)
    src_corpus = read_all_tokens(sys.argv[1])
    trg_corpus = read_all_tokens(sys.argv[2])
    assert len(src_corpus) == len(trg_corpus), "Corpora should be same size!"
    counts = count_word_cooccurrences(src_corpus, trg_corpus)
    alignments = align_corpus(src_corpus, trg_corpus, counts)
    for alignment in alignments:
        print(" ".join(["%d-%d-*" % (src, trg) for src, trg in alignment.items()]))

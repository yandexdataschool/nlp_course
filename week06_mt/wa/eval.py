#!/usr/bin/python
import sys, codecs

def parse_aligned_token(aligned_token):
    # parse a token consisting of the form '0-1-S'.
    parts = aligned_token.split('-')
    src_index, trg_index, kind = int(parts[0]), int(parts[1]), parts[2]
    return src_index, trg_index, kind

def parse_alignments(alignment_path):
    # read all alignments from file
    alignments = []
    for line in open(alignment_path):
        these_alignments = {}
        for aligned_token in line.strip().split():
            src_index, trg_index, kind = parse_aligned_token(aligned_token)
            if trg_index not in these_alignments:
                these_alignments[trg_index] = {}
            assert src_index not in these_alignments[trg_index]
            these_alignments[trg_index][src_index] = kind
        alignments.append(these_alignments)
    return alignments

def validate(src_corpus, trg_corpus, alignments):
    # check all alignment points are valid given corpora.
    assert len(src_corpus) == len(trg_corpus)
    assert len(src_corpus) == len(alignments), "%d != %d" % (
        len(src_corpus), len(alignments))
    for i in range(len(src_corpus)):
        for trg_index in alignments[i]:
            assert trg_index >= 0
            assert trg_index < len(trg_corpus[i]), '%s %d' % (trg_corpus[i], trg_index)
            for src_index in alignments[i][trg_index]:
                assert src_index >= 0
                assert src_index < len(src_corpus[i]), '%s %d' % (src_corpus[i], src_index)
    return True

def recall(reference, candidate):
    # proportion of sure alignments in reference that were found
    reference_sure, candidate_sure_correct = 0, 0
    assert len(reference) == len(candidate)
    for i, ref in enumerate(reference):
        for src_index in ref:
            for trg_index in ref[src_index]:
                if ref[src_index][trg_index] == "S":
                    reference_sure += 1
                    if src_index in candidate[i]:
                        if trg_index in candidate[i][src_index]:
                            candidate_sure_correct += 1
    return reference_sure, candidate_sure_correct

def precision(reference, candidate):
    # proportion of candidate alignments that are correct
    candidate_correct_any, candidate_total = 0, 0
    assert len(reference) == len(candidate)
    for i, cand in enumerate(candidate):
        for src_index in cand:
            for trg_index in cand[src_index]:
                candidate_total += 1
                if src_index in reference[i]:
                    if trg_index in reference[i][src_index]:
                        candidate_correct_any += 1
    return candidate_total, candidate_correct_any

def score(reference, candidate):
    reference_sure, candidate_sure_correct = recall(reference, candidate)
    candidate_total, candidate_correct_any = precision(reference, candidate)
    recall_score = 0.0
    if reference_sure > 0:
        recall_score = float(candidate_sure_correct) / reference_sure
    precision_score = 0
    if candidate_total > 0:
        precision_score = float(candidate_correct_any) / candidate_total
    aer = 1.0
    if candidate_total + reference_sure > 0:
        aer = 1.0 - float(candidate_sure_correct + candidate_correct_any) / (
            candidate_total + reference_sure)
    return recall_score, precision_score, aer

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python eval.py src_corpus trg_corpus reference candidate")
        sys.exit(0)
    src_corpus = [line.strip().split() for line in codecs.open(
        sys.argv[1], 'r', 'utf8')]
    trg_corpus = [line.strip().split() for line in codecs.open(
        sys.argv[2], 'r', 'utf8')]
    reference = parse_alignments(sys.argv[3])
    candidate = parse_alignments(sys.argv[4])
    assert validate(src_corpus, trg_corpus, reference)
    assert validate(src_corpus, trg_corpus, candidate)
    print("recall %1.3f; precision %1.3f; aer %1.3f" % score(reference, candidate))

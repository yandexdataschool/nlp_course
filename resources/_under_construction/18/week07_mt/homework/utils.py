"""Various utilities for word alignment homework."""

def recall(reference, candidate):
    "Compute recall of candidate on 'sure' alignments in reference"
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
    "Compute precision of candidate on all alignments in reference."
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

def score_alignments(reference, candidate):
    "Compute the alignment error rate for candidate against reference."
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

def validate(src_corpus, trg_corpus, alignments):
    "Checks that the alignments are valid."
    assert len(src_corpus) == len(trg_corpus)
    if not alignments:
        return True
    assert len(src_corpus) == len(alignments), "%d != %d" % (
        len(src_corpus), len(alignments))
    for i in range(len(src_corpus)):
        for trg_index in alignments[i]:
            assert trg_index >= 0
            assert trg_index < len(trg_corpus[i]), '%s %d' % (len(trg_corpus[i]), trg_index)
            for src_index in alignments[i][trg_index]:
                assert src_index >= 0
                assert src_index < len(src_corpus[i]), '%s %d' % (len(src_corpus[i]), src_index)
    return True

def extract_test_set_alignments(aligned_corpus, test_sets={"dev" : (0, 800), "test" : (800, 1600), "blinds" : (1600, 2500)}):
    "Extracts word alignments for test sets stored at start of training data ('en-cs.all')"
    test_set_alignments = {}
    for test, (start, end) in test_sets.items():
        if len(aligned_corpus) < end:
            print("skipping test set %s." % test)
            continue
        test_set_alignments[test] = [alignment for src, trg, alignment in aligned_corpus[start:end]]
    return test_set_alignments

def parse_aligned_token(aligned_token):
    parts = aligned_token.split('-')
    src_index, trg_index, kind = int(parts[0]), int(parts[1]), parts[2]
    return src_index, trg_index, kind

def read_parallel_corpus(path, has_alignments=False):
    "Load a parallel corpus from disk and tokenize."
    max_src_length, max_trg_length = 0, 0
    src_corpus, trg_corpus, alignments = [], [], [] if has_alignments else None
    for line in open(path):
        if has_alignments:
            src, trg, aligned = line.strip().split('\t')
        else:
            src, trg = line.strip().split('\t')
        src_corpus.append(src.split())
        trg_corpus.append(trg.split())
        if has_alignments:
            these_alignments = {}
            for aligned_token in aligned.strip().split():
                src_index, trg_index, kind = parse_aligned_token(aligned_token)
                if trg_index not in these_alignments:
                    these_alignments[trg_index] = {}
                assert src_index not in these_alignments[trg_index]
                these_alignments[trg_index][src_index] = kind
            alignments.append(these_alignments)
    assert validate(src_corpus, trg_corpus, alignments)
    return src_corpus, trg_corpus, alignments

def alignment_string(alignments):
    align_strings = []
    for src_index, trg_indices in alignments.items():
        for trg_index, kind in trg_indices.items():
            align_strings.append('%d-%d-%s' % (src_index, trg_index, kind))
    return ' '.join(align_strings)

def write_aligned_corpus(aligned_corpus, suffix, src_lang='en', trg_lang='cs'):
    out = open('%s-%s-wa.%s' % (src_lang, trg_lang, suffix), 'w')
    for src_tokens, trg_tokens, alignments in aligned_corpus:
        out.write('%s\t%s\t%s\n' % (' '.join(src_tokens), ' '.join(trg_tokens), alignment_string(alignments)))

import codecs

def output_alignments_per_test_set(alignments, output_prefix, test_sets={"blinds" : (0, 171), "dev" : (171, 341), "test" : (341, 511)}):
    "Outputs word alignments for test sets: this assumes corpus was from start of .all file"
    for test, (start, end) in test_sets.items():
        if len(alignments) < end:
            print("skipping test set %s." % test)
            continue
        output = open(output_prefix + "." + test + ".wa", "w")
        for alignment in alignments[start:end]:
            output.write(" ".join(["%d-%d-*" % (src_index, trg_index)
                                   for trg_index, src_index in enumerate(alignment)]) + "\n")

def read_all_tokens(path):
    corpus = []
    for line in open(path):
        corpus.append(line.strip().split())
    return corpus

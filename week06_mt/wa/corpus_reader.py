import os, sys, codecs

def read_all_tokens(path):
    return [line.strip().split() for line in codecs.open(path, 'r', 'utf8')]

def parse_alignment(aligned_token):
    parts = aligned_token.split('-')
    assert len(parts) == 3, aligned_token
    return int(parts[0]), int(parts[1]), parts[2]

def read_alignments(path):
    alignments = {}
    for sent_index, line in enumerate(open(path)):
        if sent_index not in alignments:
            alignments[sent_index] = {}
        for aligned_token in line.strip().split():
            src_index, trg_index, kind = parse_alignment(aligned_token)
            if trg_index not in alignments[sent_index]:
                alignments[sent_index][trg_index] = {}
            if src_index not in alignments[sent_index][trg_index]:
                alignments[sent_index][trg_index][src_index] = []
            alignments[sent_index][trg_index][src_index].append(kind)
    return alignments

def validate(src_corpus, trg_corpus, alignments):
    assert len(src_corpus) == len(trg_corpus)
    assert len(src_corpus) == len(alignments)
    for i in range(len(src_corpus)):
        for trg_index in alignments[i]:
            assert trg_index >= 0
            assert trg_index < len(trg_corpus[i])
            for src_index in alignments[i][trg_index]:
                assert src_index >= 0
                assert src_index < len(src_corpus[i]), "sent: %s, %s %s" % (i, src_index, trg_index)
    return True

class CorpusBrowser:

    def __init__(self, src_path, trg_path, wa_path):
        self.src_corpus_ = read_all_tokens(src_path)
        self.trg_corpus_ = read_all_tokens(trg_path)
        self.alignments_ = read_alignments(wa_path)
        assert validate(self.src_corpus_, self.trg_corpus_, self.alignments_)
        self.sent_index_, self.src_index_, self.trg_index_, self.window_, self.token_size_ = (
            0, 0, 0, 10, 5)
        self.RefreshDisplay()

    def Truncate(self, token):
        truncated = token[:self.token_size_]
        return truncated + ' ' * (self.token_size_ - len(truncated))

    def GetAlignment(self, src_index, trg_index):
        if self.sent_index_ in self.alignments_:
            if trg_index in self.alignments_[self.sent_index_]:
                if src_index in self.alignments_[self.sent_index_][trg_index]:
                    return ''.join(self.alignments_[self.sent_index_][trg_index][src_index])
        return ''

    def RefreshDisplay(self):
        print('\n' * 100)
        src_end = min([self.src_index_ + self.window_, len(self.src_corpus_[self.sent_index_])])
        src_tokens = self.src_corpus_[self.sent_index_][self.src_index_:src_end]
        src_line = ''.join([self.Truncate(src_tok) + '|' for src_tok in [''] + src_tokens])
        print(src_line)
        print('-' * len(src_line))
        trg_end = min([self.trg_index_ + self.window_, len(self.trg_corpus_[self.sent_index_])])
        trg_tokens = self.trg_corpus_[self.sent_index_][self.trg_index_:trg_end]
        for j, trg_index in enumerate(range(self.trg_index_, trg_end)):
            line = self.Truncate(trg_tokens[j]) + "|"
            for src_index in range(self.src_index_, src_end):
                kind = self.GetAlignment(src_index, trg_index)
                line += kind + ' ' * (self.token_size_ - len(kind)) + '|'
            print(line)
            print('-' * len(line))
        print()
        print(' '.join(self.src_corpus_[self.sent_index_]))
        print(' '.join(self.trg_corpus_[self.sent_index_]))
        print()
        print('n next sentence; p previous sentence')
        print('> scroll source right; < scroll source left')
        print('k scroll target right; m scroll target left')
        print('W increase window; w decrease window')
        print('T increase token size; t decrease token size')
        print('q quit')

    def HandleInput(self, input):
        if input in 'qQ':
            return False
        if input in 'nN':
            self.sent_index_ += 1
            if self.sent_index_ == len(self.src_corpus_):
                self.sent_index_ = 0
        if input in 'pP':
            self.sent_index_ -= 1
            if self.sent_index_ == -1:
                self.sent_index_ = len(self.src_corpus_) - 1
        if input in '>,':
            if self.src_index_ + self.window_ < len(self.src_corpus_[self.sent_index_]):
                self.src_index_ += 1
        if input in '<.':
            if self.src_index_ > 0:
                self.src_index_ -= 1
        if input in 'mM':
            if self.trg_index_ + self.window_ < len(self.trg_corpus_[self.sent_index_]):
                self.trg_index_ += 1
        if input in 'kK':
            if self.trg_index_ > 0:
                self.trg_index_ -= 1
        if input in 'W':
            self.window_ += 1
        if input in 'w':
            if self.window_ > 0:
                self.window_ -= 1
        if input in 'T':
            self.token_size_ += 1
        if input in 't':
            if self.token_size_ > 0:
                self.token_size_ -= 1
        return True

if __name__ == "__main__":
    if not len(sys.argv) == 4:
        print("Usage: python corpus_browser.py src_corpus trg_corpus word_alignments")
        sys.exit(0)
    browser = CorpusBrowser(sys.argv[1], sys.argv[2], sys.argv[3])
    while browser.HandleInput(input()):
        browser.RefreshDisplay()

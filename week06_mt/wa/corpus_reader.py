import os, sys, codecs
from utils import read_parallel_corpus

class CorpusBrowser:

    def __init__(self, path):
        self.src_corpus_, self.trg_corpus_, self.alignments_ = read_parallel_corpus(path, has_alignments=True)
        self.sent_index_, self.src_index_, self.trg_index_, self.window_, self.token_size_ = (
            0, 0, 0, 10, 5)
        self.RefreshDisplay()

    def Truncate(self, token):
        truncated = token[:self.token_size_]
        return truncated + ' ' * (self.token_size_ - len(truncated))

    def GetAlignment(self, src_index, trg_index):
        if src_index in self.alignments_[self.sent_index_]:
            if trg_index in self.alignments_[self.sent_index_][src_index]:
                return ''.join(self.alignments_[self.sent_index_][src_index][trg_index])
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
    if not len(sys.argv) == 2:
        print("Usage: python corpus_browser.py path.")
        sys.exit(0)
    browser = CorpusBrowser(sys.argv[1])
    while browser.HandleInput(input()):
        browser.RefreshDisplay()

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, WordPunctTokenizer
tokenizer = WordPunctTokenizer()
tokenize = lambda sent: ' '.join(tokenizer.tokenize(sent.lower()))

def build_dataset(path, tokenized=False, verbose=False, min_chars_sent=10,
                  random_state=42, test_size=0.25,
                  prune_impossible=True,  prune_all_correct=True):
    '''
    :param path: path to SQuAD data, e.g. https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
    :param tokenized: if True, data is lowercased and processed with WordPunctTokenizer
    :param verbose: if True, displays progressbar
    :returns: [train, test] - each is a dataframe with (question, possible_answers, correct_answer_indices) as columns
    '''
    data = json.load(open(path))['data']
    dataset = []
    for row in (tqdm(data) if verbose else data):
        for paragraph_id, paragraph in enumerate(row['paragraphs']):
            current_sent, current_index = '', 0
            sentences, offsets = [], []
            for raw_sent in sent_tokenize(paragraph['context']):
                current_sent += ' ' + raw_sent
                if len(current_sent) > min_chars_sent:
                    current_index += len(current_sent)
                    sentences.append(current_sent.strip())
                    offsets.append(current_index)
                    current_sent = ''
                    
            if len(current_sent):
                sentences.append(current_sent.strip())
                offsets.append(len(paragraph['context']))
                current_sent = ''
            
            if tokenized:
                sentences_tok = list(map(tokenize, sentences))
            
            for qa in paragraph['qas']:
                question = qa['question']
                correct_indices = set()
                for answer in qa['answers']:
                    #find a sentence that contains an answer
                    for i, offset, sent in zip(range(len(offsets)), offsets, sentences):
                        if answer['answer_start'] < offset:
                            correct_indices.add(i)
                            break
                    else:
                        raise ValueError('error: correct answer not found')
                
                if tokenized:
                    question_tok = tokenize(question)
                if prune_impossible and not len(correct_indices): 
                    continue
                if prune_all_correct and len(sentences) <= len(correct_indices):
                    continue
                    
                wrong_indices = [i for i in range(len(sentences)) if i not in correct_indices]
                dataset.append((paragraph_id,
                                question_tok if tokenized else question,
                                sentences_tok if tokenized else sentences, 
                                sorted(correct_indices), sorted(wrong_indices)))
                
    data = pd.DataFrame(dataset, columns=['paragraph_id', 'question', 'options', 'correct_indices', 'wrong_indices'])
    train_pid, test_pid = map(set, train_test_split(data['paragraph_id'].unique(), test_size=test_size,
                                                    random_state=random_state))
    is_train = data['paragraph_id'].apply(lambda pid: pid in train_pid)
    return data.loc[is_train], data.loc[~is_train]

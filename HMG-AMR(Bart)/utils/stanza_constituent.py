import stanza
# stanza.download('en')

import pickle
import spacy
import re
from spacy.language import Language
# Processing English text

# print(type(en_doc))


# file_name = "PycharmCode/python-process/data/dev_sentences.txt"
# file_name = "PycharmCode/python-process/data/train_sentences.txt"



class GENERATE_CONSTITUENT:
    def __init__(
            self,
            this_tokenizer,
            input_file_dir,
            pickle_dir,
    ):
        self.tokenizer = this_tokenizer
        self.data_list = self.read_data_list(input_file_dir)
        self.split_sentence()
        self.all_result_list = self.generate_cons()
        self.write_constituent(pickle_dir)


    def read_data_list(self, input_file_dir):
        self.data_list = []
        with open(input_file_dir) as file:
            for line in file:
                line.strip("\n")
                self.data_list.append(line)
        return self.data_list

    @Language.component("set_custom_boundaries")
    def set_custom_boundaries(doc):
        boundary = re.compile('^[0-9]$')
        prev = doc[0].text
        length = len(doc)
        for index, token in enumerate(doc):
            if (token.text == '.' and boundary.match(prev) and index != (length - 1)):
                doc[index + 1].sent_start = False
            prev = token.text
        return doc

    def split_sentence(self):
        nlp = spacy.load("en_core_web_sm")
        # nlp.add_pipe(self.custom_seg, before='parser')
        nlp.add_pipe("set_custom_boundaries", before="parser")
        segment_sentence_list = []
        for parapraph_sample in self.data_list:
            parapraph_sample_segment = ""
            doc = nlp(parapraph_sample)
            for sent in doc.sents:
                parapraph_sample_segment = parapraph_sample_segment + str(sent) + "\n"
            segment_sentence_list.append(parapraph_sample_segment[:-1])
        self.segment_sentence_list = segment_sentence_list



    def generate_cons(self):
        # constituency parsing
        # en_nlp = stanza.Pipeline('en', processors='tokenize, mwt, pos, constituency', tokenize_pretokenized=True)
        en_nlp = stanza.Pipeline('en')
        self.all_result_list = []
        for i, data in enumerate(self.data_list):
            data = "I love-hh my parents. I love my wife!"
            en_doc = en_nlp(data)
            paragraph_list = []
            for index, sentence in enumerate(en_doc.sentences):
                line_parseTree = sentence.constituency
                parseTree = process_tree(line_parseTree)
                paragraph_list.append(parseTree)
                if len(en_doc.sentences) - 1 == index:
                    self.all_result_list.append(paragraph_list)
                print(sentence.constituency)

    def write_constituent(self,pickle_dir):
        with open(pickle_dir, 'wb') as f:
            pickle.dump(self.all_result_list, f)




class ParseTreeNode:
    def __init__(self, val):
        self.value = val
        self.children = []
        self.height = 0
        self.parent = None
        # attributes for generating S and hierarchical positional embeddings
        self.leaf_order_idx = -1  # 0 indexed
        self.leaf_list = []

    def add_child(self, child):
        self.children.append(child)
        child.parent = self


def process_tree(root):
    current = ParseTreeNode(root.label)
    for ch in root.children:
        child = process_tree(ch)
        current.add_child(child)
        current.height = max(current.height, child.height + 1)
        child.parent = current
    return current


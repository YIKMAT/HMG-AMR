
import pickle
from collections import deque
from spring_amr.parse_tree import ParseTreeNode

class MultiPhrase():
    def __init__(self, sentence_tree_path,tokenizer, subtree_cut_height = 4):
        self.sentence_tree_path = sentence_tree_path
        self.subtree_cut_height = subtree_cut_height
        self.tokenizer = tokenizer
        self.max_phrases_count = -1
        self.max_phrases_len = -1
        self.read_sentence_tree()
        # -----选择：按句子按短语，还是只按短语
        # self.generate_phrase()
        self.generate_phrase_without_paragraph()
        #------生成整个句子
        # self.generate_sentence_without_paragraph()




    # 将句子的短语结构树读进来
    def read_sentence_tree(self):
        tree_list = []
        with open(self.sentence_tree_path,'rb') as f:
            tree_list = pickle.load(f)
        self.tree_list = tree_list


    # 生成不同的phrase list 【考虑paragraph中的多个句子，按句子按短语输出】
    def generate_phrase(self):
        self.phrases_list, self.phrases_len_list = [], []
        for paragraph_tree in self.tree_list:
            paragraph_phrases_list = []
            paragraph_phrases_len_list = []
            for tree in paragraph_tree:
                self.build_leaf_paths(tree)
                phrases, phrases_len = self.getTreePhrases(tree, self.subtree_cut_height)
                paragraph_phrases_list.append(phrases)
                paragraph_phrases_len_list.append(phrases_len)
                self.max_phrases_count = max(self.max_phrases_count, len(phrases))
                self.max_phrases_len = max(self.max_phrases_len, max(phrases_len))
            self.phrases_list.append(paragraph_phrases_list)
            self.phrases_len_list.append(paragraph_phrases_len_list)

    # 生成不同的phrase list 【不考虑paragraph中的多个句子，只按短语输出】
    def generate_phrase_without_paragraph(self):
        self.phrases_list, self.phrases_len_list = [], []
        for paragraph_tree in self.tree_list:
            phrase_temp = []
            prase_len_temp = []
            for index, tree in enumerate(paragraph_tree):
                self.build_leaf_paths(tree)
                phrases, phrases_len = self.getTreePhrases(tree, self.subtree_cut_height)
                self.max_phrases_count = max(self.max_phrases_count, len(phrases))
                self.max_phrases_len = max(self.max_phrases_len, max(phrases_len))
                if len(paragraph_tree) == 1:
                    # -----给短语首尾加入<Ġ<s>> 和 <Ġ<s>>,，如果不需要，就去掉--------
                    phrases.insert(0,["Ġ<s>"])
                    phrases.append(["Ġ</s>"])
                    phrases_len.insert(0,1)
                    phrases_len.append(1)
                    self.phrases_list.append(phrases)
                    self.phrases_len_list.append(phrases_len)

                else:
                    for in_phrase in phrases:
                        phrase_temp.append(in_phrase)
                        prase_len_temp.append(len(in_phrase))
                if len(paragraph_tree) > 1 and index == len(paragraph_tree)-1:
                    # -----给短语首尾加入<Ġ<s>> 和 <Ġ<s>>,，如果不需要，就去掉--------
                    phrase_temp.insert(0,["Ġ<s>"])
                    phrase_temp.append(["Ġ</s>"])
                    prase_len_temp.insert(0,1)
                    prase_len_temp.append(1)
                    self.phrases_list.append(phrase_temp)
                    self.phrases_len_list.append(prase_len_temp)

    # 生成不同的sentence_list 【不考虑paragraph中的多个句子，只按短语输出】
    def generate_phrase_without_paragraph(self):
        self.phrases_list, self.phrases_len_list = [], []
        for paragraph_tree in self.tree_list:
            phrase_temp = []
            prase_len_temp = []
            for index, tree in enumerate(paragraph_tree):
                self.build_leaf_paths(tree)
                phrases, phrases_len = self.getTreePhrases(tree, self.subtree_cut_height)
                self.max_phrases_count = max(self.max_phrases_count, len(phrases))
                self.max_phrases_len = max(self.max_phrases_len, max(phrases_len))
                if len(paragraph_tree) == 1:
                    # -----给短语首尾加入<Ġ<s>> 和 <Ġ<s>>,，如果不需要，就去掉--------
                    phrases.insert(0,["Ġ<s>"])
                    phrases.append(["Ġ</s>"])
                    phrases_len.insert(0,1)
                    phrases_len.append(1)
                    self.phrases_list.append(phrases)
                    self.phrases_len_list.append(phrases_len)

                else:
                    for in_phrase in phrases:
                        phrase_temp.append(in_phrase)
                        prase_len_temp.append(len(in_phrase))
                if len(paragraph_tree) > 1 and index == len(paragraph_tree)-1:
                    # -----给短语首尾加入<Ġ<s>> 和 <Ġ<s>>,，如果不需要，就去掉--------
                    phrase_temp.insert(0,["Ġ<s>"])
                    phrase_temp.append(["Ġ</s>"])
                    prase_len_temp.insert(0,1)
                    prase_len_temp.append(1)
                    self.phrases_list.append(phrase_temp)
                    self.phrases_len_list.append(prase_len_temp)


    def build_leaf_paths(self,current_node):
        '''
        Stores all leaves for each internal node of tree
        '''
        #store all leaves in subtree rooted at current_node in order
        if current_node.height==0: #is leaf
            current_node.leaf_list.append(current_node)

        for child in current_node.children:
            self.build_leaf_paths(child)
            current_node.leaf_list += child.leaf_list


    def getTreePhrases(self,node,cut_height = 4, method = 'dfs'):
        '''
        Get phrases from tree
        '''
        phrases,phrases_len = [],[]
        Q = deque()
        Q.append(node)
        while len(Q)>0:
            if method=='dfs':
                top = Q.pop()
            elif method=='bfs':
                top = Q.popleft()
            if top.height <= cut_height:
                phrase = [leaf_node.value for leaf_node in top.leaf_list]
                # -------------------这里调用Bart的tokenizer，提前切分成子词--------------------
                # 如不需要子词，则可在此处直接注释这一行
                phrase = self.tokenizer.tokenize(" ".join(phrase), add_special_tokens=False)
                phrases.append(phrase)
                phrases_len.append(len(phrase))
            else:
                if method=='dfs':
                    iteration_list = reversed(top.children)
                elif method=='bfs':
                    iteration_list = top.children
                for child in iteration_list:
                    Q.append(child)
        return phrases,phrases_len
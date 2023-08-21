
class SentenceSplit():
    def __init__(self, sentence_path,tokenizer):
        self.sentence_path = sentence_path
        self.tokenizer = tokenizer
        self.read_sentence_tree()
        self.generate_sentence_len()

    # 将句子文件读进来
    def read_sentence_tree(self):
        self.sentence_list = []
        with open(self.sentence_path,'r') as f:
            line_list = f.readlines()
        all_sentence_list = []
        sample = []
        for index, line in enumerate(line_list):
            if line != "\n":
                sample.append(line.strip())
            if line == "\n" or index == len(line_list)-1:
                all_sentence_list.append(sample)
                sample = []
        self.sentence_list = all_sentence_list

    # 生成句子长度文件
    def generate_sentence_len(self):
        self.multi_sentence_num = 0
        sentence_len_list = []
        for sentences in self.sentence_list:
            sentences_len = []
            for index, sentence in enumerate(sentences):
                sentence_token = self.tokenizer.tokenize(sentence, add_special_tokens=False)
                if len(sentences) == 1:
                    # 加上首尾的<s>和</s>
                    sentences_len.append(len(sentence_token)+2)
                    sentence_len_list.append(sentences_len)
                    continue
                if len(sentences) > 1 and index == 0:
                    # 加上首的<s>
                    sentences_len.append(len(sentence_token) + 1)
                    continue
                elif len(sentences) > 1 and index == len(sentences)-1:
                    # 加上尾的</s>
                    sentences_len.append(len(sentence_token) + 1)
                    sentence_len_list.append(sentences_len)
                    self.multi_sentence_num = self.multi_sentence_num + 1
                    continue
                else:
                    sentences_len.append(len(sentence_token))

        self.sentence_len_list = sentence_len_list

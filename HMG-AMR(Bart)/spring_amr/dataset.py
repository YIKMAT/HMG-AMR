import logging
import random
import torch
from cached_property import cached_property
from torch.utils.data import Dataset
from spring_amr.IO import read_raw_amr_data
from spring_amr.multi_phrase import MultiPhrase
from spring_amr.multi_clause import MultiClause
from spring_amr.sentence_split import SentenceSplit
import copy
import numpy as np

def reverse_direction(x, y, pad_token_id=1):
    input_ids = torch.cat([y['decoder_input_ids'], y['lm_labels'][:, -1:]], 1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == pad_token_id] = 0
    decoder_input_ids = x['input_ids'][:,:-1]
    lm_labels = x['input_ids'][:,1:]
    x = {'input_ids': input_ids, 'attention_mask': attention_mask}
    y = {'decoder_input_ids': decoder_input_ids, 'lm_labels': lm_labels}
    return x, y

class AMRDataset(Dataset):
    
    def __init__(
        self,
        paths,
        tokenizer,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
            p_mg=None,
        constituent_file =None,
            sentence_file = None,
            clause_file=None
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.device = device
        graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        self.mg_phrase_1_len = []
        self.mg_phrase_2_len = []
        self.mg_phrase_3_len = []
        self.mg_subsentence_len = []
        self.mg_sentence_len = []
        self.mg_sentence = []
        # -------------------短语输入--------------
        p_mg = [int(num) for num in p_mg.split("-")]
        mg_phrase_1_len = MultiPhrase(constituent_file, self.tokenizer, p_mg[0])
        mg_phrase_2_len = MultiPhrase(constituent_file, self.tokenizer, p_mg[1])
        mg_phrase_3_len = MultiPhrase(constituent_file, self.tokenizer, p_mg[2])
        # -------------------子句输入--------------
        mg_subsentence_len = MultiClause(clause_file, self.tokenizer)
        # -------------------句子--------------
        mg_sentence_len = SentenceSplit(sentence_file, self.tokenizer)
        # self.mg_sentence_len = MultiPhrase(constituent_file, self.tokenizer, 5)

        for index, g in enumerate(graphs):
            l, e = self.tokenizer.linearize(g)
            # if e['graphs'].metadata['id'] == "bolt-eng-DF-170-181103-8887658_0014.12":
            #     print("daole")
            # l是该图的线性化表示（经过bpe的，ids表示）。e里面包含"'linearized_graphs"(该图的线性化表示，单词表示)和"graphs"（原图的三元组等）.
            try:
                self.tokenizer.batch_encode_sentences([g.metadata['snt']])
            except:
                logging.warning('Invalid sentence!')
                continue

            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')

            # =====================根据是否需要计算该样本，这里逐个加入===============
            # -------------------短语输入--------------
            self.mg_phrase_1_len.append(mg_phrase_1_len.phrases_len_list[index])
            self.mg_phrase_2_len.append(mg_phrase_2_len.phrases_len_list[index])
            self.mg_phrase_3_len.append(mg_phrase_3_len.phrases_len_list[index])
            # -------------------子句输入--------------
            self.mg_subsentence_len.append(mg_subsentence_len.phrases_len_list[index])
            # -------------------句子--------------
            self.mg_sentence_len.append(mg_sentence_len.sentence_len_list[index])
            self.mg_sentence.append(mg_sentence_len.sentence_list[index])

            self.sentences.append(g.metadata['snt'])
            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)
        # ==================句子的难度=====================
        self.difficulty = self.get_difficulty()

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]

        # ------------------phrase输入-----------------------
        sample['mg_phrase_1_len'] = self.mg_phrase_1_len[idx]
        sample['mg_phrase_2_len'] = self.mg_phrase_2_len[idx]
        sample['mg_phrase_3_len'] = self.mg_phrase_3_len[idx]
        # ==================clause输入=========================
        sample['mg_subsentence_len'] = self.mg_subsentence_len[idx]
        # ==================sentence输入=========================
        sample['mg_sentence_len'] = self.mg_sentence_len[idx]
        sample['mg_sentences'] = self.mg_sentence[idx]


        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])            
        return sample
    #  在sampler(self)时，取出一个样本后，测量其长度
    def size(self, sample):
        return len(sample['linearized_graphs_ids'])
    
    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        # x, extra = self.tokenizer.batch_encode_sentences(x, device=device)
        # ------------------phrase输入-----------------------
        mg_phrase_1_len = [s['mg_phrase_1_len'] for s in samples]
        mg_phrase_2_len = [s['mg_phrase_2_len'] for s in samples]
        mg_phrase_3_len = [s['mg_phrase_3_len'] for s in samples]
        # ==================clause输入=========================
        mg_subsentence_len = [s['mg_subsentence_len'] for s in samples]
        # ==============句子====================
        mg_sentences = [s['mg_sentences'] for s in samples]
        mg_sentence_len = [s['mg_sentence_len'] for s in samples]

        x, extra = self.tokenizer.batch_encode_sentences_multi(x, copy.deepcopy(mg_phrase_1_len), copy.deepcopy(mg_phrase_2_len),
                                        copy.deepcopy(mg_phrase_3_len), copy.deepcopy(mg_sentence_len), copy.deepcopy(mg_subsentence_len), copy.deepcopy(mg_sentences), device=device)
        # phrase_list = [s['phrase_list'] for s in samples]
        # phrase_list_len = [s['phrase_list_len'] for s in samples]
        # x, extra = self.tokenizer.batch_encode_sentences_phrase(x, copy.deepcopy(phrase_list), copy.deepcopy(phrase_list_len), device=device)
        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra

    def get_difficulty(self):
        num_clauses = [len(sublist) for sublist in self.mg_subsentence_len]
        num_sentences = [len(sublist) for sublist in self.mg_sentence_len]
        sum = [a+b for a,b in zip(num_clauses,num_sentences)]
        diffcults = np.array(sum) - min(sum)
        return diffcults
    
class AMRDatasetTokenBatcherAndLoader:
    
    def __init__(self, dataset, batch_size=800 ,device=torch.device('cpu'), shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    def __iter__(self):
        it = self.sampler()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    @cached_property
    def sort_ids(self):
        lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    # 在    def __iter__(self)函数的 it = ([[self.dataset[s] for s in b] for b in it])调用一次，按照batch_size大小，
    # 遍历所有样本，将几个样本长度之和 小于 batch_size的条件，将样本分为很多个batch，一个样本只分一次！
    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]
        # 训练的时候，shuffle；测试的时候，sort；
        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()

class CurriculumAMRDatasetTokenBatcherAndLoader:

    def __init__(self, dataset, batch_size=800 ,device=torch.device('cpu'), shuffle=False, sort=False, IC_steps=500):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort
        self.IC_steps = IC_steps
        self.diffcults = self.dataset.difficulty
        print(f'[MAX_layer] {self.diffcults.max()}; [IC_steps]: {IC_steps}')

    def __iter__(self):
        it = self.sampler_curriculum()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it


    def sampler_curriculum(self):
        max_layer = self.diffcults.max()
        for layer in range(max_layer+1):
            for _ in range(self.IC_steps):
                mask = self.diffcults <= layer
                optional_indexs = np.array(range(0, len(self.dataset)))[mask]
                ids = np.random.choice(optional_indexs, replace=False, size=100).tolist() # 选100个句子作为该batch候选
                # ids = np.random.choice(optional_indexs, replace=False, size=1).tolist() # 选100个句子作为该batch候选

                batch_longest = 0
                batch_nexamps = 0
                batch_ntokens = 0
                batch_ids = []

                def discharge():
                    nonlocal batch_longest
                    nonlocal batch_nexamps
                    nonlocal batch_ntokens
                    ret = batch_ids.copy()
                    batch_longest *= 0
                    batch_nexamps *= 0
                    batch_ntokens *= 0
                    batch_ids[:] = []
                    return ret
                while batch_ntokens < self.batch_size and len(ids) > 0:
                    idx = ids.pop()
                    size = self.dataset.size(self.dataset[idx])
                    cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
                    if cand_batch_ntokens > self.batch_size and batch_ids:
                        break
                    batch_longest = max(batch_longest, size)
                    batch_nexamps += 1
                    batch_ntokens = batch_longest * batch_nexamps
                    batch_ids.append(idx)
                yield discharge()
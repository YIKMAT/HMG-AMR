
class MultiClause():
    def __init__(self, all_clause_path,tokenizer):
        self.all_clause_path = all_clause_path
        self.tokenizer = tokenizer
        self.max_phrases_count = -1
        self.max_phrases_len = -1
        self.all_sample_list = []
        # 从子句文件中，读取出相应的信息,形成列表
        self.read_clause_tree()
        # 提取出子句信息
        self.get_clause_info()



    """
    函数作用：一行一行的读入clause文件，提取出相应的信息。
    输入：子句文件
    输出：所有样本的list。[sentence, tokens, index, dfs_clauses]
    """
    def get_clause(self, input_f):
        indicate_string = ""
        sample_list = []
        for line in input_f:
            if line == "\n":
                # end of current AMR
                break
            if line.startswith("# ::sentence"):
                sent = line.strip("# ::sentence").strip()
                sample_list.append(sent)
            if line.startswith("# ::tokens"):
                ids = line.strip("# ::tokens").strip()
                sample_list.append(ids)
            if line.startswith("# ::index"):
                sent = line.strip("# ::index").strip()
                sample_list.append(sent)
            if line.startswith("# ::dfs_clauses"):
                sent = line.strip("# ::dfs_clauses").strip()
                sample_list.append(sent)
            indicate_string += line

        return indicate_string, sample_list

    """
    函数作用：从子句文件中，读取出相应的信息。
    输入：子句文件
    输出：所有样本的list。[sentence, tokens, index, dfs_clauses]
    """
    def read_clause_tree(self):
        f = open(self.all_clause_path)
        while True:
            amr, sample_list = self.get_clause(f)
            if amr == "":
                break
            self.all_sample_list.append(sample_list)
        print("样本数量："+str(len(self.all_sample_list)))



    # def get_same_split(self, indexs):
    #     result = []
    #     current_group = [indexs[0]]
    #
    #     for i in range(1, len(indexs)):
    #         if indexs[i] != indexs[i - 1]:
    #             result.append(current_group)
    #             current_group = []
    #         current_group.append(indexs[i])
    #
    #     # 添加最后一个分组
    #     result.append(current_group)
    #
    #     return result

    """
       函数作用：根据子句index，将token切分成子句
       输入：tokens, indices
       输出：切分好的子句token
       """
    def get_same_split_tokens(self, tokens, indices):
        result = []
        current_group = []
        current_index = indices[0]

        for i in range(len(tokens)):
            if indices[i] != current_index:
                result.append(current_group)
                current_group = []
                current_index = indices[i]
            current_group.append(tokens[i])
        # 添加最后一个分组
        result.append(current_group)

        return result

    """
    函数作用：从子句信息的list中，返回出相应的信息。
    输入：子句信息的list
    输出：self.phrases_list, self.phrases_len_list, self.dfs_clauses_list
    """
    def get_clause_info(self):
        self.phrases_list, self.phrases_len_list, self.dfs_clauses_list = [], [], []
        for sample in self.all_sample_list:
        # 将token和index变为list
            tokens = sample[1].split(" ")
            indices = sample[2].split(" ")
            clauses = []
            clauses_len = []
            # =====简单句=====
            if indices == ["-1"]:
                clause = self.tokenizer.tokenize(sample[0], add_special_tokens=False)
                # -----给clause首尾加入<Ġ<s>> 和 <Ġ<s>>,，如果不需要，就去掉--------
                clause.insert(0, "Ġ<s>")
                clause.append("Ġ</s>")
                clauses.append(clause)
                clauses_len.append(len(clause))
            else:
                same_split_tokens = self.get_same_split_tokens(tokens, indices)
                for i, clause_token in enumerate(same_split_tokens):
                    clause = self.tokenizer.tokenize(" ".join(clause_token), add_special_tokens=False)
                    # -----给clause首尾加入<Ġ<s>> 和 <Ġ<s>>,，如果不需要，就去掉--------
                    if i == 0:
                        clause.insert(0, "Ġ<s>")
                    if i == len(same_split_tokens)-1:
                        clause.append("Ġ</s>")
                    clauses.append(clause)
                    clauses_len.append(len(clause))

            #=====这里使用 phrases_list和phrases_len_list命名，是因为后续便于统一计算
            self.phrases_list.append(clauses)
            self.phrases_len_list.append(clauses_len)
            self.dfs_clauses_list.append(sample[3])




if __name__ == '__main__':
    MultiClause("/sda/yikmat/PycharmCode/Hierar_AMR/HMG-AMR-2/data/AMR/clause/train_min_clause_all.txt")



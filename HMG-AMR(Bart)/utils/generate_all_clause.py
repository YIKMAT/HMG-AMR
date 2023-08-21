
clause_long_filepath ="/sda/yikmat/PycharmCode/Hierar_AMR/HMG-AMR-2/data/AMR/amr_2.0/clauses/train.txt"
sen_ids_filepath ="/sda/yikmat/PycharmCode/Hierar_AMR/HMG-AMR-2/data/AMR/amr_2.0/clauses/train_sen_ids.txt"
output_file = "/sda/yikmat/PycharmCode/Hierar_AMR/HMG-AMR-2/data/AMR/amr_2.0/clauses/AMR2.0_train_clause_all.txt"

clause_long_dict = {}
sen_ids_list = []
all_sample_list = []

"""
函数的作用：输入所有样例的sen_ids文件（ids,sentence,tokens）, 以及复杂句的clause文件。输出所有样例的clause文件（ids,sentence,tokens,index,relation）。
输入：所有样例的sen_ids文件（ids,sentence,tokens）, 以及复杂句的clause文件
输出： 所有样例的clause文件（ids,sentence,tokens,index,relation）


输出样例：
# ::id bolt12_07_4800.2
# ::sentence After its competitor invented the front loading washing machine, the CEO of the American IM company believed that each of its employees had the ability for innovation , and formulated strategic countermeasures for innovation in the industry.
# ::tokens After its competitor invented the front loading washing machine , the CEO of the American IM company believed that each of its employees had the ability for innovation , and formulated strategic countermeasures for innovation in the industry .
# ::index 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# ::dfs_clauses <Snt> 0 <OBJ> 1 </OBJ> <TME> 2 </TME> </Snt>

"""

# clause_long_dict key:ids value:token,index,relation
def get_clause_long_dict(input_f):

    amr_string = ""
    ids = ""
    info_list = []
    for line in input_f:
        if line == "\n":
            # end of current AMR
            break
        if line.startswith("# ::id"):
            ids = line.strip("# ::id").strip()
        if line.startswith("# ::tokens"):
            tokens = line.strip("# ::tokens").strip()
            info_list.append(tokens)
        if line.startswith("# ::index"):
            index = line.strip("# ::index").strip()
            info_list.append(index)
        if line.startswith("# ::dfs_clauses"):
            dfs_clauses = line.strip("# ::dfs_clauses").strip()
            info_list.append(dfs_clauses)
        amr_string += line

    if ids!="":
        clause_long_dict[ids] = info_list
    return amr_string


# sen_ids_list：ids, sentences, tokens
def get_sen_ids_list(input_f):

    amr_string = ""
    ids = ""
    sample_list = []
    for line in input_f:
        if line == "\n":
            # end of current AMR
            break
        if line.startswith("ids:: "):
            sample_list.append(line.strip("ids:: ").strip())
        if line.startswith("sentence:: "):
            sample_list.append(line.strip("sentence:: ").strip())
        if line.startswith("tokens:: "):
            sample_list.append(line.strip("tokens:: ").strip())

        amr_string += line


    return amr_string, sample_list

# sen_ids_list：ids, sentences, tokens
# clause_long_dict key:ids value:token,index,relation
# 目标：ids, sentences, tokens, index, relation
def generate_result():
    result = ""
    for sample in sen_ids_list:
        ids = sample[0]
        clause_info = clause_long_dict.get(ids)
        if clause_info == None:
            index = "-1"
            dfs_clauses = "-1"
            sample.append(index)
            sample.append(dfs_clauses)
            result = result + '# ::id ' + sample[0] + '\n' + '# ::sentence ' + sample[
                1] + '\n' + '# ::tokens ' + sample[2] + '\n'+ '# ::index ' + sample[3] + '\n'+ '# ::dfs_clauses ' + sample[4] + '\n\n'
        else:
            if len(clause_info[0].split()) != len(sample[2].split()):
                print("这句不相等,ids:"+ids)
                # ====如果两者的tokenization不相等，则使用clause_info中的分词结果=====
                sample[2] = clause_info[0]
            sample.append(clause_info[1])
            sample.append(clause_info[2])
            result = result + '# ::id ' + sample[0] + '\n' + '# ::sentence ' + sample[
                1] + '\n' + '# ::tokens ' + sample[2] + '\n'+ '# ::index ' + sample[3] + '\n'+ '# ::dfs_clauses ' + sample[4] + '\n\n'
        all_sample_list.append(sample)
    return result


def write_sent(result):
    f = open(output_file, "w")
    f.write(result)
    # for line in sent_list:
    #     f.write(line + '\n')
    f.close()


def generate_clause_all(clause_long_filepath):
    # line = f.readline()
    # while line:
    #     print (line)
    #     print(type(line))
    #     line = f.readline()
    # f.close()
    # （1）将clause_long文件的内容读进来，形成dict，以便匹配
    f = open(clause_long_filepath)
    sum_num = 0
    sent_list = []
    while True:
        amr_string = get_clause_long_dict(f)
        if amr_string == "":
            break
        sum_num += 1
    if sum_num != len(clause_long_dict):
        print("字典与输入不符")
    # （2）将sen_ids文件的内容读进来，形成list，以便匹配
    sum_num = 0
    f = open(sen_ids_filepath)
    while True:
        amr_string, sample_list = get_sen_ids_list(f)
        if amr_string == "":
            break
        sum_num += 1
        sen_ids_list.append(sample_list)
    if sum_num != len(sen_ids_list):
        print("句子和id数量不符")
    # （3）合成所有的，并导出
    result_string = generate_result()
    write_sent(result_string)
    return sum_num












if __name__ == '__main__':

    sum_num = generate_clause_all(clause_long_filepath)
    print("总数是: ",sum_num)
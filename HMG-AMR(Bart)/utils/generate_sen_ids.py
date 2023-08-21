import stanza
en_nlp = stanza.Pipeline('en',processors='tokenize,pos,lemma')
filepath ="/sda/yikmat/PycharmCode/Hierar_AMR/HMG-AMR-2/data/AMR/clause/train_min.txt"
sent_file = "/sda/yikmat/PycharmCode/Hierar_AMR/HMG-AMR-2/data/AMR/clause/train_min_sen_ids.txt"


"""
函数的作用：输入amr原始数据集, 生成带有ids,sentence,tokens的文件。
输入：amr原始数据集.
输出： 带有ids, sentence, tokens的文件.

输出样例：
ids:: bolt12_07_4800.1
sentence:: Establishing Models in Industrial Innovation.
tokens:: Establishing Models in Industrial Innovation .
"""

def get_amr_string(input_f):
    """
    Read the file containing AMRs. AMRs are separated by a blank line.
    Each call of get_amr_line() returns the next available AMR (in one-line form).
    Note: this function does not verify if the AMR is valid

    """
    amr_string = ""
    ids = ""
    sent = ""
    sample_list = []
    for line in input_f:
        if line == "\n":
            # end of current AMR
            break
        if line.startswith("# ::id"):
            ids = line.split('::date')[0].strip("# ::id").strip()
            sample_list.append(ids)
        if line.startswith("# ::snt"):
            sent = line.strip("# ::snt").strip()
            sample_list.append(sent)
        amr_string += line

    return amr_string, sent, sample_list


def get_token(all_sample_list):
    result = ""
    for sample in all_sample_list:
        en_doc = en_nlp(sample[1])
        token_list = []
        for sentence in en_doc.sentences:
            token_list += [word.text for word in sentence.words]
        sentence_token = " ".join(token_list)
        result = result + 'ids:: '+ sample[0] + '\n'+'sentence:: '+ sample[1] + '\n'+'tokens:: '+sentence_token + '\n\n'
    return result

def write_sent(result):
    f = open(sent_file, "w")
    f.write(result)
    # for line in sent_list:
    #     f.write(line + '\n')
    f.close()


def generate_sen_ids(filepath):
    # line = f.readline()
    # while line:
    #     print (line)
    #     print(type(line))
    #     line = f.readline()
    # f.close()
    f = open(filepath)
    sum_num = 0
    sent_list = []
    all_sample_list = []
    while True:
        amr, sent, sample_list = get_amr_string(f)
        if amr == "":
            break
        sum_num += 1
        sent_list.append(sent)
        all_sample_list.append(sample_list)
    if sum_num != len(sent_list):
        print("图与句子不符")
    # 得到token=====
    result = get_token(all_sample_list)
    write_sent(result)
    return sum_num


if __name__ == '__main__':

    sum_num = generate_sen_ids(filepath)
    print("总数是: ",sum_num)


# def get_amr_string(input_f):
#     """
#     Read the file containing AMRs. AMRs are separated by a blank line.
#     Each call of get_amr_line() returns the next available AMR (in one-line form).
#     Note: this function does not verify if the AMR is valid
#
#     """
#     amr_string = ""
#     ids = ""
#     sent = ""
#     sample_list = []
#     for line in input_f:
#         if line == "\n":
#             # end of current AMR
#             break
#         if line.startswith("# ::id"):
#             ids = line.split('::date')[0].strip("# ::id").strip()
#             sample_list.append(ids)
#         if line.startswith("# ::snt"):
#             sent = line.strip("# ::snt").strip()
#             sample_list.append(sent)
#         amr_string += line
#
#     return amr_string, sent, sample_list
#
#
# def get_token(all_sample_list):
#     result = ""
#     for sample in all_sample_list:
#         en_doc = en_nlp(sample)
#         for sentence in en_doc.sentences:
#             token_list = [word.text for word in sentence.words]
#             sentence_token = " ".join(token_list)
#             result = result + 'ids::'+ sample[0] + '\n'+'sentence::'+ sample[1] + '\n'+'tokens::'+sentence_token + '\n\n'
#     return result
#
# def write_sent(sent_list):
#     f = open(sent_file, "w")
#
#     for line in sent_list:
#         f.write(line + '\n')
#     f.close()
#
#
# def generate_sen_ids(filepath):
#     # line = f.readline()
#     # while line:
#     #     print (line)
#     #     print(type(line))
#     #     line = f.readline()
#     # f.close()
#     f = open(filepath)
#     sum_num = 0
#     sent_list = []
#     all_sample_list = []
#     while True:
#         amr, sent, sample_list = get_amr_string(f)
#         if amr == "":
#             break
#         sum_num += 1
#         sent_list.append(sent)
#         all_sample_list.append(sample_list)
#     if sum_num != len(sent_list):
#         print("图与句子不符")
#     # 得到token=====
#     result = get_token(all_sample_list)
#     write_sent(sent_list)
#     return sum_num
#
#
# if __name__ == '__main__':
#
#     sum_num = generate_sen_ids(filepath)
#     print("总数是: ",sum_num)
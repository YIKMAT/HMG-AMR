import torch
from transformers import TensorType

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_data(batch_text, tokenizer, max_length, index, mg_dict, mg_list, pad_id=None, return_overflowing_tokens=True):
    """
    Prepare datasets
    :param batch_text: a list of source sequences ["I like monkey", "How are you"]
    :param tokenizer: tokenizer
    :param max_length: maximum source models length
    :param pad_id: pad_id
    :param return_overflowing_tokens: if true return overflowing tokens
    :return: a dict with sequence_ids and sequence_attention_mask
    """
    input_encodings = tokenizer.batch_encode_plus(batch_text,
                                                  padding=True,
                                                  truncation=True if max_length is not None else False,
                                                  max_length=max_length,
                                                  return_attention_mask=True,
                                                  # return_overflowing_tokens=return_overflowing_tokens,
                                                  return_tensors=TensorType.PYTORCH)

    # Convert to tensors
    input_ids = torch.LongTensor(input_encodings['input_ids']).to(device)
    attention_mask = torch.LongTensor(input_encodings['attention_mask']).to(device)

    mg_dict_batch = {}
    # =====组装资源信息========
    if mg_list is not None:
        # =========短语=========
        if mg_list[0] != 0:
            temp_list = []
            for sample_index in index:
                temp_list.append(mg_dict["mg_phrase_1_len"][sample_index])
            mg_dict_batch["mg_phrase_1_len"] = temp_list
        if mg_list[1] != 0:
            temp_list = []
            for sample_index in index:
                temp_list.append(mg_dict["mg_phrase_2_len"][sample_index])
            mg_dict_batch["mg_phrase_2_len"] = temp_list
        if mg_list[2] != 0:
            temp_list = []
            for sample_index in index:
                temp_list.append(mg_dict["mg_phrase_3_len"][sample_index])
            mg_dict_batch["mg_phrase_3_len"] = temp_list
        # batch["mg_phrase_3_len"] = mg_phrase_3_len
        # ===========子句=========
        if mg_list[3] != 0:
            temp_list = []
            for sample_index in index:
                temp_list.append(mg_dict["mg_subsentence_len"][sample_index])
            mg_dict_batch["mg_subsentence_len"] = temp_list
        # =========句子=======这个一定要放到最后
        temp_list = []
        for sample_index in index:
            temp_list.append(mg_dict["mg_sentence_len"][sample_index])
            mg_dict_batch["mg_sentence_len"] = temp_list

    if pad_id is not None:
        mask = input_ids == tokenizer.pad_token_id
        input_ids[mask] = pad_id

    return input_ids, attention_mask, mg_dict_batch


def prepare_data_original(batch_text, tokenizer, max_length, pad_id=None, return_overflowing_tokens=True):
    """
    Prepare datasets
    :param batch_text: a list of source sequences ["I like monkey", "How are you"]
    :param tokenizer: tokenizer
    :param max_length: maximum source models length
    :param pad_id: pad_id
    :param return_overflowing_tokens: if true return overflowing tokens
    :return: a dict with sequence_ids and sequence_attention_mask
    """
    input_encodings = tokenizer.batch_encode_plus(batch_text,
                                                  padding=True,
                                                  truncation=True if max_length is not None else False,
                                                  max_length=max_length,
                                                  return_attention_mask=True,
                                                  # return_overflowing_tokens=return_overflowing_tokens,
                                                  return_tensors=TensorType.PYTORCH)

    # Convert to tensors
    input_ids = torch.LongTensor(input_encodings['input_ids']).to(device)
    attention_mask = torch.LongTensor(input_encodings['attention_mask']).to(device)

    if pad_id is not None:
        mask = input_ids == tokenizer.pad_token_id
        input_ids[mask] = pad_id

    return input_ids, attention_mask

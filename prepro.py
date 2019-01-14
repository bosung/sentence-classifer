import ujson as json
from tqdm import tqdm
from collections import Counter
import numpy as np
from codecs import open
import spacy
from nltk.tokenize import sent_tokenize

nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def sentence_tokenize(context):
    sent_list = list()
    minimum_next = 0
    for i, raw_sent in enumerate(sent_tokenize(context)):
        start_idx = context.find(raw_sent, minimum_next)
        if raw_sent[0] != context[start_idx]:
            print(raw_sent)
            print(start_idx, context)
            raise IOError
        sent_tokens = word_tokenize(raw_sent)
        end_idx = start_idx + len(raw_sent) - 1
        sent_list.append([(start_idx, end_idx), raw_sent, sent_tokens])
        # add 2 for last space.
        minimum_next = end_idx + 1
    return sent_list


def parse_file(raw_file, word_counter):
    """ parse raw json file and make train file """
    print("[INFO] parsing %s..." % raw_file)
    data = []
    with open(raw_file, "r") as f:
        source = json.load(f)
        articles = source["data"]
        for article in articles:
            paragraphs = article["paragraphs"]
            for p in paragraphs:
                context = p["context"]

                context_sents = sentence_tokenize(context)
                for (_, _, sent_tokens) in context_sents:
                    for t in sent_tokens:
                        word_counter[t] += 1

                qas = p["qas"]
                for qa in qas:
                    q = qa["question"]
                    # answers = qa["answers"]  # list type of dict('text', 'answer_start')
                    _id = qa["id"]
                    is_impossible = qa["is_impossible"]
                    if is_impossible:
                        answers = qa["plausible_answers"]
                    else:
                        answers = qa["answers"]

                    q_tokens = word_tokenize(q)
                    for t in q_tokens:
                        word_counter[t] += 1

                    for (idx, raw_sent, sent_tokens) in context_sents:
                        data.append({"question": q,
                                     "q_tokens": q_tokens,
                                     "id": _id,
                                     "answers": answers,
                                     "is_impossible": is_impossible,
                                     "context_raw_sent": raw_sent,
                                     "context_sent_tokens": sent_tokens,
                                     "context_idx": idx})

    print("[DONE] parsing %s" % raw_file)
    return data


def build_sent_sim(config):
    """ parse raw json file and make train file for sentence similarity """
    obj_file = config.raw_json_train_file
    # obj_file = config.raw_json_dev_file
    print("[INFO] parsing %s..." % obj_file)
    data = []
    with open(obj_file, "r") as f:
        source = json.load(f)
        articles = source["data"]
        for article in articles:
            paragraphs = article["paragraphs"]
            for p in paragraphs:
                context = p["context"]
                context_sents = sentence_tokenize(context)

                qas = p["qas"]
                for qa in qas:
                    q = qa["question"]
                    _id = qa["id"]
                    is_impossible = qa["is_impossible"]
                    if is_impossible:
                        answers = qa["plausible_answers"]
                    else:
                        answers = qa["answers"]

                    answer_idx = []
                    for ans in answers:
                        answer_start = int(ans["answer_start"])
                        for i, (idx, raw_sent, sent_tokens) in enumerate(context_sents):
                            si, ei = idx
                            if si <= answer_start <= ei:
                                answer_idx.append(i)

                    data.append({"question": q,
                                 "q_tokens": word_tokenize(q),
                                 "id": _id,
                                 "answers": answers,
                                 "is_impossible": is_impossible,
                                 "context_sent_list": context_sents,
                                 "answer_idx": answer_idx})

    print("[DONE] parsing %s" % obj_file)
    return data


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {} {}...".format(len(obj), message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def get_embedding(counter, data_type, emb_file=None, size=None, vec_size=None):
    print("Generating %s embedding..." % data_type)
    embedding_dict = {}
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(embedding_dict), data_type))
    else:
        assert vec_size is not None
        for token in counter:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(len(counter)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def get_word(word2idx_dict, word):
    if word == 0:
        return 1
    for each in (word, word.lower(), word.capitalize(), word.upper()):
        if each in word2idx_dict:
            return word2idx_dict[each]
    return 1


def char2idx(word, c_max_length):
    char_idx = list()
    for c in word:
        asc_code = ord(c)
        # None -> 0, 'A' -> 1, 'a' -> 'Z'+ 1
        if ord('A') <= asc_code <= ord('Z'):
            char_idx.append(asc_code - ord('A') + 1)
        elif ord('a') <= asc_code <= ord('z'):
            char_idx.append(asc_code - 70)
        else:
            char_idx.append(0)

    if len(word) < c_max_length:
        # padding
        for _ in range(c_max_length - len(word)):
            char_idx.append(0)
        return char_idx
    else:
        return char_idx[:c_max_length]


def char_convert_with_padding(tokens, max_length, c_max_length=7):
    result = list()
    for word in tokens:
        result.append(char2idx(word, c_max_length))

    if len(tokens) < max_length:
        # padding
        for _ in range(max_length - len(tokens)):
            result.append([0] * c_max_length)
        return result
    else:
        return result[:max_length]


def convert2idx_with_padding(tokens, w2i_dict, _max_length):
    result = list()
    for _token in tokens:
        result.append(get_word(w2i_dict, _token))

    if len(tokens) < _max_length:
        # padding
        for _ in range(_max_length - len(tokens)):
            result.append(0)
        return result
    else:
        return result[:_max_length]


def build_features(data, w2i_dict, config):
    q_max_length = config.q_max_length
    c_sent_max_length = config.c_sent_max_length

    for d in data:
        d["question_idx"] = convert2idx_with_padding(d["q_tokens"], w2i_dict, q_max_length)
        d["question_char_idx"] = char_convert_with_padding(d["q_tokens"], q_max_length)
        if "context_sent_tokens" in d:
            d["sentence_idx"] = convert2idx_with_padding(d["context_sent_tokens"], w2i_dict, c_sent_max_length)
            d["sentence_char_idx"] = char_convert_with_padding(d["context_sent_tokens"], c_sent_max_length)
            for ans in d["answers"]:
                answer_start = int(ans["answer_start"])
                answer_end = answer_start + len(ans['text']) - 1
                sent_start, sent_end = int(d["context_idx"][0]), int(d["context_idx"][1])
                if sent_start <= answer_start <= sent_end:
                    d["label_idx"] = 1
                    break
                elif sent_start <= answer_end <= sent_end:
                    d["label_idx"] = 1
                    break
                else:
                    d["label_idx"] = 0
        if "context_sent_list" in d:
            d["context_sent_list_idx"] = []
            for i, (idx, raw_sent, sent_tokens) in enumerate(d["context_sent_list"]):
                d["context_sent_list_idx"].append(convert2idx_with_padding(sent_tokens, w2i_dict, c_sent_max_length))


def build_transfer_data(raw_file, w2i_dict, config):
    q_max_length = config.q_max_length
    c_sent_max_length = config.c_sent_max_length

    with open(raw_file, "r") as f:
        data = json.load(f)
        for d in data:
            question_tokens = word_tokenize(d["question"])
            context_tokens = word_tokenize(d["filtered_context"])
            d["question_idx"] = convert2idx_with_padding(question_tokens, w2i_dict, q_max_length)
            d["context_idx"] = convert2idx_with_padding(context_tokens, w2i_dict, c_sent_max_length)
            d["question_char_idx"] = char_convert_with_padding(question_tokens, config.q_max_length)
            d["context_char_idx"] = char_convert_with_padding(context_tokens, config.c_sent_max_length)

        return data


def prepro(config):
    """
    word_counter, char_counter = Counter(), Counter()

    train_data = parse_file(config.raw_json_train_file, word_counter)
    dev_data = parse_file(config.raw_json_dev_file, word_counter)
    test_data = parse_file(config.raw_json_test_file, word_counter)

    f = open(config.word2idx_file, 'r')
    word2idx_dict = json.load(f)
    # word_emb_mat, word2idx_dict = get_embedding(word_counter, "word", emb_file=config.glove_word_file,
    #    size=config.glove_word_size, vec_size=config.glove_dim)

    build_features(train_data, word2idx_dict, config)
    build_features(dev_data, word2idx_dict, config)
    build_features(test_data, word2idx_dict, config)
    
    # save(config.word_emb_file, word_emb_mat, message="word embedding")
    # save(config.word2idx_file, word2idx_dict, message="word2idx")
    save(config.train_file, train_data, message="train data")
    save(config.dev_file, dev_data, message="dev data")
    save(config.test_file, test_data, message="test data")
    """
    # sentence sim-eval
    f = open(config.word2idx_file, 'r')
    word2idx_dict = json.load(f)
    sim_eval = build_sent_sim(config)
    build_features(sim_eval, word2idx_dict, config)
    save(config.sim_eval_test, sim_eval, message="sentence similarity test file")
    """
    # transfer
    # f = open(config.word2idx_file, 'r')
    # word2idx_dict = json.load(f)
    transfer_train = build_transfer_data(config.transfer_train_file, word2idx_dict, config)
    transfer_dev = build_transfer_data(config.transfer_dev_file, word2idx_dict, config)

    save(config.transfer_train_file, transfer_train, message="transfer train file")
    save(config.transfer_dev_file, transfer_dev, message="transfer dev file")
    """

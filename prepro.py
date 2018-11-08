import ujson as json
from tqdm import tqdm
from collections import Counter
import numpy as np
from codecs import open
import spacy

nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def sent_tokenize(context):
    sent_list = []
    start_idx = 0
    next_period = context.find('. ')
    end_idx = next_period
    while next_period > 0:
        sent_list.append([(start_idx, end_idx+1), context[start_idx:end_idx+1]])
        start_idx = end_idx+2
        next_period = context[start_idx:].find('. ')
        end_idx = start_idx + next_period
    sent_list.append([(start_idx, len(context)), context[start_idx:]])
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
                context_sents = sent_tokenize(context)

                for (_, sent) in context_sents:
                    tokens = word_tokenize(sent)
                    for t in tokens:
                        word_counter[t] += 1

                qas = p["qas"]
                for qa in qas:
                    q = qa["question"]
                    # answers = qa["answers"]  # list type of dict('text', 'answer_start')
                    is_impossible = qa["is_impossible"]
                    if is_impossible:
                        answers = qa["plausible_answers"]
                    else:
                        answers = qa["answers"]

                    q_tokens = word_tokenize(q)
                    for t in q_tokens:
                        word_counter[t] += 1

                    for (idx, sent) in context_sents:
                        data.append({"question": q,
                                     "answers": answers,
                                     "is_impossible": is_impossible,
                                     "context_sent": sent,
                                     "context_idx": idx})

    print("[DONE] parsing %s" % raw_file)
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


def build_features(data, word2idx_dict, config):
    q_max_length = config.q_max_length
    c_sent_max_length = config.c_sent_max_length

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _convert2idx_with_padding(tokens, _max_length):
        result = list()
        for _token in tokens:
            result.append(_get_word(_token))

        if len(tokens) < _max_length:
            # padding
            for _ in range(_max_length - len(tokens)):
                result.append(0)
            return result
        else:
            return result[:_max_length]

    for d in data:
        d["question_idx"] = _convert2idx_with_padding(d["question"], q_max_length)
        d["sentence_idx"] = _convert2idx_with_padding(d["context_sent"], c_sent_max_length)
        for ans in d["answers"]:
            answer_start = int(ans["answer_start"])
            sent_start, sent_end = int(d["context_idx"][0]), int(d["context_idx"][1])
            if sent_start <= answer_start <= sent_end:
                d["label_idx"] = 1
                break
            else:
                d["label_idx"] = 0


def prepro(config):
    word_counter, char_counter = Counter(), Counter()

    train_data = parse_file(config.raw_json_train_file, word_counter)
    dev_data = parse_file(config.raw_json_dev_file, word_counter)
    test_data = parse_file(config.raw_json_test_file, word_counter)

    word_emb_mat, word2idx_dict = get_embedding(word_counter, "word", emb_file=config.glove_word_file,
        size=config.glove_word_size, vec_size=config.glove_dim)

    build_features(train_data, word2idx_dict, config)
    build_features(dev_data, word2idx_dict, config)
    build_features(test_data, word2idx_dict, config)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.train_file, train_data, message="train data")
    save(config.dev_file, dev_data, message="dev data")
    save(config.test_file, test_data, message="test data")

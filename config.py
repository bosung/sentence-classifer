import os


class Config:
    model_name = "base"
    # data
    home = os.getcwd()
    data_dir = os.path.join(home, "data")
    # raw_json_train_file = os.path.join(home, "datasets", "squad", "train-v2.0.json")
    # raw_json_dev_file = os.path.join(home, "datasets", "squad", "squad_2.0_output_1.json")
    # raw_json_test_file = os.path.join(home, "datasets", "squad", "squad_2.0_output_2.json")
    raw_json_train_file = os.path.join("/home/nlp908", "data", "squad", "train-v2.0.json")
    raw_json_dev_file = os.path.join("/home/nlp908", "data", "squad", "squad_2.0_output_1.json")
    raw_json_test_file = os.path.join("/home/nlp908", "data", "squad", "squad_2.0_output_2.json")
    train_file = os.path.join(home, "data", "train.json")
    dev_file = os.path.join(home, "data", "dev.json")
    test_file = os.path.join(home, "data", "test.json")

    # sentence similarity expr
    sim_eval_test = os.path.join(data_dir, "sim-eval.json")

    # word embedding
    word_emb_file = os.path.join(data_dir, "word_emb.json")
    word2idx_file = os.path.join(data_dir, "word2idx.json")

    # word embedding
    glove_word_file = os.path.join("/home", "nlp908", "data", "glove", "glove.840B.300d.txt")
    # glove_word_file = os.path.join(home, "datasets", "glove.840B.300d.txt")
    glove_word_size = int(2.2e6)
    glove_dim = 300

    # hyper param
    q_max_length = 15
    c_sent_max_length = 30
    batch_size = 64
    test_batch_size = 1
    epoch = 10
    learning_rate = 0.001
    # optim = "SGD"
    optim = "RMSprop"
    # optim = "Adam"

    # train
    checkpoint = 10
    state_dict_dir = os.path.join(home, "state_dict")
    state_dict = os.path.join(state_dict_dir, model_name, "checkpoint-")

    # test
    test_model = os.path.join(state_dict_dir, model_name, "checkpoint-3")

    def __init__(self):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        if not os.path.exists(self.state_dict_dir):
            os.mkdir(self.state_dict_dir)
        if not os.path.exists(self.state_dict_dir + '/' + self.model_name):
            os.mkdir(self.state_dict_dir + '/' + self.model_name)

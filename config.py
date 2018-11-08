import os


class Config:
    # data
    home = os.getcwd()
    data_dir = os.path.join(home, "data")
    raw_json_train_file = os.path.join(home, "datasets", "squad", "train-v2.0.json")
    raw_json_dev_file = os.path.join(home, "datasets", "squad", "squad_2.0_output_1.json")
    raw_json_test_file = os.path.join(home, "datasets", "squad", "squad_2.0_output_2.json")
    train_file = os.path.join(home, "data", "train.json")
    dev_file = os.path.join(home, "data", "dev.json")
    test_file = os.path.join(home, "data", "test.json")

    word_emb_file = os.path.join(data_dir, "word_emb.json")

    # word embedding
    # glove_word_file = os.path.join("/home", "nlpgpu4", "data", "glove", "glove.840B.300d.txt")
    glove_word_file = os.path.join(home, "datasets", "glove.840B.300d.txt")
    glove_word_size = int(2.2e6)
    glove_dim = 300

    # hyper param
    q_max_length = 15
    c_sent_max_length = 30
    batch_size = 64
    epoch = 10
    learning_rate = 0.1
    optim = "SGD"
    # optim = "RMSprop"
    # optim = "Adam"

    # train
    checkpoint = 10
    state_dict_dir = os.path.join(home, "state_dict")
    state_dict = os.path.join(state_dict_dir, "checkpoint-")

    # test
    test_model = os.path.join(state_dict_dir, "checkpoint-19")

    def __init__(self):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        if not os.path.exists(self.state_dict_dir):
            os.mkdir(self.state_dict_dir)

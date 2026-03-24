from configs import config


class LorenzTimeInvariantConfig(config.Config):
    def __init__(self):
        super(LorenzTimeInvariantConfig, self).__init__()

        self.MODEL_NAME = 'lorenz'

        self.USE_CNN_MLP = True
        self.CNN_FILTERS = [64, 32]

        self.DROP_RATE = 0

        self.EMBEDDING_LEN = 38

        self.K = 9

        self.TRAIN_LEN = 96

        self.SPATIAL_NODES=[256,self.K]

        self.MERGE_MAP_NODES = [512, 256, 128, self.EMBEDDING_LEN]

        self.Y_IDX = 0

        self.MODULE_LAST_ACITVATION = False

        self.limit_cases = False

        self.DATASET_RATE = 1

        self.TRAINING = True  # whther is training

        self.ENCODING_LAYER_NUMS = 1  # the stack number of self-attention

        self.TEMPORAL_DIM = self.K


        self.SPARTICAL_DIM = self.TRAIN_LEN
        self.DIFF = self.SPARTICAL_DIM * 4

        self.NUM_HEADS = 3

        self.BATCH_SIZE = 4

        self.EPOCHS = 100

        self.LOSS_WEIGHTS = {'consistent_loss': 1,}

        self.ENCODER_DILATION_RATES = [1, 2, 4, 4, 8]

        self.ENCODER_NODES = [128, 128, 64, 64, self.K]


        self.TCN_BLOCK_REPEAT_TIMES = 1

        self.TCN_BLOCK_RESIDUAL = False

        self.KERNEL_SIZE = 3

        self.KERNEL_REGULARIZER = 'l2'

        self.KERNEL_INITIALIZER = 'he_normal'

        self.NORMALIZATION = 'ln'

        self.MERGE_ONLY = False

        self.MERGE_FUNC = 'film'

        self.LR = 1e-3

        self.WEIGHT_DECAY = 5e-3

        self.iterations_count = 10

        self.BN = False

        self.ADD_NOISE = False

        self.DATA_NOISE_STRENGTH = 0

        if self.MERGE_ONLY:

            self.MODEL_NAME = self.MODEL_NAME + '_no_TemporalModel'

    def LR_SCHEDULER(self, epoch):
        if epoch <= 20:
            return self.LR
        elif epoch <= 40:
            return self.LR / 3.0
        elif epoch <= 65:
            return self.LR / 10.0
        else:
            return self.LR / 30.0
    @property
    def name(self):
        if self.ADD_NOISE:
            return '{}_{}_{}_Yidx_{}_noise_{}'.format(self.MODEL_NAME, self.TRAIN_LEN, self.EMBEDDING_LEN, self.Y_IDX, self.DATA_NOISE_STRENGTH)
        elif self.limit_cases:
            return '{}_{}_{}_Yidx_{}_rate_{}'.format(self.MODEL_NAME, self.TRAIN_LEN, self.EMBEDDING_LEN, self.Y_IDX, self.DATASET_RATE)
        else:
            return '{}_{}_{}_Yidx_{}'.format(self.MODEL_NAME, self.TRAIN_LEN, self.EMBEDDING_LEN, self.Y_IDX)

from configs import config


class KSConfig(config.Config):
    def __init__(self):
        super(KSConfig, self).__init__()

        self.MODEL_NAME = 'KS'  # the name of data

        self.DROP_RATE = 0

        self.EMBEDDING_LEN = 28

        self.K = 512

        self.TRAIN_LEN = 96

        self.USE_CNN_MLP = True
        self.CNN_FILTERS = [8, 4]

        self.SPATIAL_NODES = [512, 256, 256]

        self.MERGE_MAP_NODES = [512, 256, 128, self.EMBEDDING_LEN]

        self.Y_IDX = 6

        self.MODULE_LAST_ACITVATION = False

        self.DATASET_RATE = 0

        self.TRAINING = True  

        self.ENCODING_LAYER_NUMS = 2 

        self.TEMPORAL_DIM = self.K 

        self.DIFF = self.TEMPORAL_DIM * 4

        self.NUM_HEADS = 16

        self.BATCH_SIZE = 4

        self.EPOCHS = 100

        self.LOSS_WEIGHTS = {'consistent_loss': 1}

        self.ENCODER_DILATION_RATES = [1, 2, 4, 4, 8]

        self.ENCODER_NODES = [512, 256, 128, 128, 256]

        self.TCN_BLOCK_REPEAT_TIMES = 1

        self.TCN_BLOCK_RESIDUAL = False

        self.KERNEL_SIZE = 3

        self.KERNEL_REGULARIZER = 'l2'

        self.KERNEL_INITIALIZER = 'he_normal'

        self.NORMALIZATION = 'ln'

        self.MERGE_ONLY = False  

        self.MERGE_FUNC = 'film'  # 'add'

        self.LR = 1e-4

        self.WEIGHT_DECAY = 1e-3

        self.BN = False

        self.ADD_NOISE =  True # False

        self.DATA_NOISE_STRENGTH = 0.3

        if self.MERGE_ONLY:
            self.MODEL_NAME = self.MODEL_NAME + '_merge_only'

    def LR_SCHEDULER(self, epoch):
        if epoch <= 10:
            return self.LR
        elif epoch <= 40:
            return self.LR / 3.0
        elif epoch <= 60:
            return self.LR / 10.0
        elif epoch <= 90:
            return self.LR / 30.0
        else:
            return self.LR / 100.0


    @property
    def name(self):
        if self.ADD_NOISE:
            return '{}_{}_{}_Yidx_{}_noise_{}'.format(self.MODEL_NAME, self.TRAIN_LEN, self.EMBEDDING_LEN, self.Y_IDX, self.DATA_NOISE_STRENGTH)
        else:
            return '{}_{}_{}_Yidx_{}'.format(self.MODEL_NAME, self.TRAIN_LEN, self.EMBEDDING_LEN, self.Y_IDX)

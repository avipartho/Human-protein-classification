class DefaultConfigs(object):
    # DIRECTORIES
    train_data = "../human_atlas/data/train/" # train data directory
    test_data = "../human_atlas/data/test/"   # test data directory
    data_root = "../human_atlas/data/"		  # data root directory
    weights = "./checkpoints/"                # saved models' directory
    best_models = "./checkpoints/best_models/"# best models' directory
    submit = "./submit/"                      # submission file directory

    model_name = "bninception_bcelog4"
    # model_name = "bninception_focalLoss"
    # model_name = "bninception_f1Loss"

    # PARAMETERS
    num_classes = 28
    img_width = 512
    img_height = 512
    channels = 4
    lr = 0.03
    batch_size = 32
    epochs = 50
    curr_fold = 0 
    gpu_no = "1"

    # FLAGS
    img_normalized = False
    first_layer_pretrained = True
    weighted_loss = False
    oversample = True
    
    train = True
    retrain = True # train flag should be True for retrain to work
    test = True
    ensemble = False

config = DefaultConfigs()
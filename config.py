JOINT = {
    'nyu':14,
    'icvl':16,
    'msra':21,
    'shrec':21,
}

STEP = {
    'nyu': 10,
    'msra': 5,
    'icvl': 4,
    'shrec':5,

}

EPOCH = {
    'nyu': 15,
    'msra': 10,
    'icvl': 8,
    'shrec':10,
}

CUBE = {
    'nyu': [250, 250, 250],
    'msra': [200, 200, 200],
    'icvl': [200, 200, 200],
     'shrec': [200, 200, 200],
}


class Config(object):
    phase = 'train'
    root_dir = '/home/pfren/dataset/hand/'# R3900/P100
    dataset = 'nyu'# ['nyu'，'icvl','msra', 'shrec']
    model_save = ''
    add_info = 'Finetune-Stage'
    train_stage = 'Finetune'
    stage_num = 2
    mask = True

    save_mesh = False
    save_result = True
    save_obj = False

    deconv_weight = 1
    coord_weight = 100
    model_weight = 1
    partICP_weight = 1
    M2P_weight = 1
    coll_weight = 1


    finetune_dir = ''
    load_model = ''
    tansferNet_pth = ''
    # finetune_dir = '/home/pfren/pycharm/hand_mixed/checkpoint/nyu/MANO-pretrain-alljoint-synth/best.pth'
    # finetune_dir = './checkpoint/nyu/Finetune-Stage-v2/latest.pth'
    # finetune_dir = './checkpoint/nyu/Pretrain-RotTransMean-xyz/best.pth'
    # finetune_dir = './checkpoint/nyu/Pretrain-Stage-NoRemap/best.pth'
    finetune_dir = './checkpoint/nyu/Pretrain-RotTransMean-xyz-2stage/best.pth'
    # tansferNet_pth = '/home/pfren/pycharm/pytorch-CycleGAN-and-pix2pix/checkpoints/ada-10/'
    # tansferNet_pth = '/home/pfren/pycharm/pytorch-CycleGAN-and-pix2pix/checkpoints/task-10/'
    # tansferNet_pth = '/home/pfren/pycharm/pytorch-CycleGAN-and-pix2pix/checkpoints/nyu_ori_cyclegan-40epoch/'
    tansferNet_pth = '/home/pfren/pycharm/pytorch-CycleGAN-and-pix2pix/checkpoints/nyu_background_consis_cyclegan-40epoch/'
    # tansferNet_pth = '/home/pfren/pycharm/pytorch-CycleGAN-and-pix2pix/checkpoints/msra_background_consis_cyclegan-40epoch/'
    # tansferNet_pth = '/home/pfren/pycharm/pytorch-CycleGAN-and-pix2pix/checkpoints/mask_consis_cyclegan-40epoch/'
    # tansferNet_pth = '/home/pfren/pycharm/pytorch-CycleGAN-and-pix2pix/checkpoints/shrec_background_consis_cyclegan-40epoch/'
    # tansferNet_pth = '/home/pfren/pycharm/pytorch-CycleGAN-and-pix2pix/checkpoints/icvl_flip_consis_cyclegan-40epoch/'
    mano_model_path = './MANO/'  # R3900

    save_dir = './'
    train_img_type = 'real'
    test_img_type = 'real'

    joint_num = JOINT[dataset]
    cube_size = CUBE[dataset]

    test_during_train = True

    batch_size = 32
    input_size = 128

    center_type = 'refine'  # ['joint_mean', 'refine']
    loss_type = 'L1Loss'
    augment_para = [10, 0.2, 180]

    lr = 0.001
    max_epoch = EPOCH[dataset]
    step_size = STEP[dataset]
    opt = 'adamw'
    scheduler = 'step'

    net = 'ResNet_stage_18'
    feature_type = ['offset']
    feature_para = [0.8]


opt = Config()


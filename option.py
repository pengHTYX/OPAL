import argparse
import os
import torch

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        option_dir = os.path.join(opt.checkpoints_dir, opt.model_name)
        if not os.path.exists(option_dir):
            os.makedirs(option_dir)
        file_name = os.path.join(option_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    
    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.is_shuffle = True

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        self.opt = opt
        return self.opt

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # basic parameters
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--output_dir', type=str, default='output', help='models are saved here')
        parser.add_argument('--time_str', type=str, default='latest', help='run date ')
        parser.add_argument('--losses', type=str, default='L1_smooth', help='loss type')
        parser.add_argument('--grad_v', type=int, default=5, help='cal occu')
        parser.add_argument('--alpha', type=float, default=1e-2, help='')
        parser.add_argument('--lamda', type=float, default=150., help='smoothness')
        parser.add_argument('--pad', type=int, default=0, help='padding')
        parser.add_argument('--debug', action='store_true', help='debug') 
        # model parameters
        parser.add_argument('--model_name', type=str, default='OPENet', help='chooses which model to use.')
        parser.add_argument('--n_block', type=int, default=6)
        parser.add_argument('--input_c', type=int, default=3) ##########
        parser.add_argument('--out_c', type=int, default=9) ##########
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # dataset parameters
        parser.add_argument('--dataset_file', type=str, default='ref41', help='[ref41 | LFattdata]')
        parser.add_argument('--root', type=str, default='./dataset', help='path of dataset')
        parser.add_argument('--use_views', type=int, default=9, help='')
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--num_val', type=int, default=100, help='val imgs')
        parser.add_argument('--max_dataset_size', type=int, default=10000, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--load_size', type=int, default=512, help='load imgs size')
        parser.add_argument('--input_size', type=int, default=64, help='input imgs size')
        parser.add_argument('--scale', type=int, default=1, help='input imgs size')
        parser.add_argument('--epoch', type=str, default='latest', help='load the latest model when test or continue_train')
        # additional parameters
        parser.add_argument('--pin_memory', action='store_true', help='')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--num_layers', type=int, default=0, help='')
        self.initialized = True
        return parser

class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=4000, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        # network saving and loading parameters
        parser.add_argument('--after_epoch_save_visual', type=int, default=100, help='frequency of saving the latest results')
        parser.add_argument('--save_latest_freq', type=int, default=8000, help='frequency of saving the latest results')  ###############
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs with the initial learning rate')              # 50
        parser.add_argument('--opti_policy', type=str, default='Adam', help='[SGD | Momentum | RMSprop | Adam ]')                                  #0.001
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate for adam')                                  #0.001
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
        
        self.isTrain = True
        return parser

class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=100, help='how many test images to run')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
       
        self.isTrain = False
        return parser

import argparse
import os
import torch
from dataloader import msasppDataLoader
from arch import SimpleAutoEncoder

parser = argparse.ArgumentParser(description='Multi-scale ASPP training')

parser.add_argument('--mode',                   type=str,   help='training and validation mode',    default='train')
parser.add_argument('--model_name',             type=str,   help='model name to be trained',        default='denseaspp-v3')

# Dataset
parser.add_argument('--data_path',              type=str,   help='training data path',              default=os.path.join(os.getcwd(), "dataset"))
parser.add_argument('--input_height',           type=int,   help='input height',                    default=512)
parser.add_argument('--input_width',            type=int,   help='input width',                     default=512)

# Training
parser.add_argument('--num_seed',               type=int,   help='random seed number',              default=1)
parser.add_argument('--batch_size',             type=int,   help='train batch size',                default=8)
parser.add_argument('--num_epochs',             type=int,   help='number of epochs',                default=80)
parser.add_argument('--learning_rate',          type=float, help='initial learning rate',           default=3e-4)
parser.add_argument('--weight_decay',           type=float, help='weight decay factor for optimization',                                default=1e-5)
parser.add_argument('--retrain',                type=bool,  help='If used with checkpoint_path, will restart training from step zero',  default=False)
parser.add_argument('--do_eval',                type=bool,  help='Mod to evaluating the training model',                                default=False)

# Preprocessing
parser.add_argument('--random_rotate',          type=bool,  help='if set, will perform random rotation for augmentation',   default=False)
parser.add_argument('--degree',                 type=float, help='random rotation maximum degree',                          default=2.5)

# Log and save
parser.add_argument('--checkpoint_path',        type=str,   help='path to a specific checkpoint to load',               default='')
parser.add_argument('--log_directory',          type=str,   help='directory to save checkpoints and summaries',         default=os.path.join(os.getcwd(), 'log'))
parser.add_argument('--log_freq',               type=int,   help='Logging frequency in global steps',                   default=250)
parser.add_argument('--save_freq',              type=int,   help='Checkpoint saving frequency in global steps',         default=500)

# Multi-gpu training
parser.add_argument('--gpu',            type=int,  help='GPU id to use', default=0)
parser.add_argument('--rank',           type=int,  help='node rank(tensor dimension)for distributed training', default=0)
parser.add_argument('--dist_url',       type=str,  help='url used to set up distributed training', default='file:///c:/MultiGPU.txt')
parser.add_argument('--dist_backend',   type=str,  help='distributed backend', default='gloo')
parser.add_argument('--num_threads',    type=int,  help='number of threads to use for data loading', default=5)
parser.add_argument('--world_size',     type=int,  help='number of nodes for distributed training', default=1)
parser.add_argument('--multiprocessing_distributed',       help='Use multi-processing distributed training to launch '
                                                                'N process per node, which has N GPUs. '
                                                                'This is the fastest way to use PyTorch for either single node or '
                                                                'multi node data parallel training', default=False)
args = parser.parse_args()

def main():
    command = 'mkdir -p ' + args.log_directory
    os.system(command)

    main_worker()
    
def main_worker():
    dataloader = msasppDataLoader(args, mode='train')

    model = SimpleAutoEncoder()
    model.train()
    model = torch.nn.DataParallel(model)
    model.cuda()

    criterion = torch.nn.BCELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)

    global_step = 0
    steps_per_epoch = len(dataloader.data)
    epoch = global_step // steps_per_epoch
    while epoch < args.num_epochs:
        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            
            sample_image = sample_batched['image'].cuda(args.gpu, non_blocking=True)
            # sample_gt = sample_batched['gt'].cuda(args.gpu, non_blocking=True)
            
            output = model(sample_image)
            # loss = criterion(output, sample_image)
            loss = torch.nn.functional.kl_div(output, sample_image) + torch.nn.functional.l1_loss(output, sample_image)
            loss.backward()
            
            for param_group in optimizer.param_groups:
                current_lr = args.learning_rate * ((1 - epoch / args.num_epochs) ** 0.9)
                param_group['lr'] = current_lr
                
            optimizer.step()
            
            # if global_step and global_step % args.log_freq == 0:
            print_string = "[epoch][s/s_per_e/global_step]: [{}/{}][{}/{}/{}] | train loss: {:.5f}"
            print(print_string.format(epoch+1, args.num_epochs, step+1, steps_per_epoch, global_step+1, loss))

            global_step += 1
        
        checkpoint = {'global_step': global_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(args.log_directory, args.model_name, 'model', 'model-{:07d}.pth'.format(global_step)))
        epoch += 1

if __name__ == "__main__":
    main()
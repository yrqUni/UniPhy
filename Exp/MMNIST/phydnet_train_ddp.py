import random
import numpy as np
import torch
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_random_seed(1017)

import sys
import os
import glob
import logging
import datetime
sys.path.append('/data/ConvLRU/Model/PhyDNet')
sys.path.append('/data/ConvLRU/DATA')

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from ModelPhyDNet import PhyCell, ConvLSTM, EncoderRNN
from constrain_moments import K2M
from MMNIST import MovingMNIST
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class Args:
    def __init__(self):
        # Model parameters
        self.phycell_input_shape = (16, 16)
        self.phycell_input_dim = 64
        self.phycell_F_hidden_dims = [49] 
        self.phycell_n_layers = 1
        self.phycell_kernel_size = (7, 7)
        self.convlstm_input_shape = (16, 16)
        self.convlstm_input_dim = 64
        self.convlstm_hidden_dims = [128,128,64]
        self.convlstm_n_layers = 3
        self.convlstm_kernel_size = (3, 3) 
        self.teacher_forcing_ratio = None
        # Data parameters
        self.data_root = '/data/ConvLRU/DATA/MMNIST/'
        self.data_train_n_frames_input = 64
        self.data_train_n_frames_output = 64
        self.data_train_num_objects = [3]
        self.data_train_num_samples = 8192
        self.data_eval_n_frames_input = 64
        self.data_eval_n_frames_output = 64
        self.data_eval_num_objects = [3]
        self.data_eval_num_samples = 512
        # Training parameters
        self.ckpt = ''
        self.train_batch_size = 2
        self.eval_batch_size = 2
        self.epochs = 2048
        self.log_path = "./exp_phydnet_0/logs"
        self.ckpt_dir = "./exp_phydnet_0/ckpt"
        self.ckpt_step = 1024
        self.eval_step = 1024

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def keep_latest_ckpts(ckpt_dir):
    ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.pth'))
    if len(ckpt_files) <= 2:
        return
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    files_to_delete = ckpt_files[2:]
    for file_path in files_to_delete:
        os.remove(file_path)

def remove_module_prefix(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

def save_ckpt(model, epoch, step, loss, args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f'e{epoch}_s{step}_l{loss:.6f}.pth'))
    keep_latest_ckpts(args.ckpt_dir)

def load_ckpt(model, ckpt_path):
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = remove_module_prefix(checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"No checkpoint Found")
    return model

def setup_logging(args):
    os.makedirs(args.log_path, exist_ok=True)
    log_filename = os.path.join(args.log_path, f'training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def log_message(message):
    logging.info(message)

def run_ddp(rank, world_size, args):
    setup_ddp(rank, world_size)
    if rank == 0:
        setup_logging(args)
    phycell  =  PhyCell(input_shape=args.phycell_input_shape, input_dim=args.phycell_input_dim, F_hidden_dims=args.phycell_F_hidden_dims, n_layers=args.phycell_n_layers, kernel_size=args.phycell_kernel_size, device=torch.device(f'cuda:{rank}'))
    convcell =  ConvLSTM(input_shape=args.convlstm_input_shape, input_dim=args.convlstm_input_dim, hidden_dims=args.convlstm_hidden_dims, n_layers=args.convlstm_n_layers, kernel_size=args.convlstm_kernel_size, device=torch.device(f'cuda:{rank}'))
    model  = EncoderRNN(phycell, convcell, torch.device(f'cuda:{rank}'))
    model = load_ckpt(model, args.ckpt).cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    loss_fn = torch.nn.MSELoss().cuda(rank)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataset = MovingMNIST(root=args.data_root, is_train=True, n_frames_input=args.data_train_n_frames_input, n_frames_output=args.data_train_n_frames_output, num_objects=args.data_train_num_objects, num_samples=args.data_train_num_samples)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)
    eval_dataset = MovingMNIST(root=args.data_root, is_train=True, n_frames_input=args.data_eval_n_frames_input, n_frames_output=args.data_eval_n_frames_output, num_objects=args.data_eval_num_objects, num_samples=args.data_eval_num_samples)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    scheduler = lr_scheduler.OneCycleLR(opt, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=args.epochs)

    model.train()

    constraints = torch.zeros((49,7,7)).cuda(rank)
    ind = 0
    for i in range(0,7):
        for j in range(0,7):
            constraints[ind,i,j] = 1
            ind +=1  

    for ep in range(args.epochs):
        train_dataloader = tqdm(train_dataloader, desc=f"Epoch {ep+1}/{args.epochs} - Start") if rank == 0 else train_dataloader
        for step, (inputs, outputs) in enumerate(train_dataloader, start=1):
            inputs, outputs = inputs.cuda(rank), outputs.cuda(rank)
            opt.zero_grad()
            loss = 0
            for ei in range(args.data_train_n_frames_input-1): 
                encoder_output, encoder_hidden, output_image,_,_ = model(inputs[:,ei,:,:,:], (ei==0))
                loss += loss_fn(output_image,inputs[:,ei+1,:,:,:])
            decoder_input = inputs[:,-1,:,:,:] 
            args.teacher_forcing_ratio = np.maximum(0 , 1 - ep * 0.003) 
            use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False 
            for di in range(args.data_train_n_frames_output):
                decoder_output, decoder_hidden, output_image,_,_ = model(decoder_input)
                target = outputs[:,di,:,:,:]
                loss += loss_fn(output_image,target)
                if use_teacher_forcing:
                    decoder_input = target   
                else:
                    decoder_input = output_image

            k2m = K2M([7,7]).cuda(rank)
            for b in range(0, model.module.phycell.cell_list[0].input_dim):
                filters = model.module.phycell.cell_list[0].F.conv1.weight[:,b,:,:]   
                m = k2m(filters.double()) 
                m  = m.float()   
                loss += loss_fn(m, constraints)
            loss.backward()
            opt.step()
            scheduler.step()

            loss_value = loss.item()
            loss_tensor = torch.tensor(loss_value).cuda(rank)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / world_size

            if step % args.eval_step == 0 and step != 0:
                model.eval()
                with torch.no_grad():
                    loss_value = 0
                    eval_dataloader = tqdm(eval_dataloader, desc=f"Eval Epoch {ep+1}/{args.epochs} - Step {step} - Start") if rank == 0 else eval_dataloader
                    for _step, (inputs, outputs) in enumerate(eval_dataloader, start=1):
                        inputs, outputs = inputs.cuda(rank), outputs.cuda(rank)
                        for ei in range(args.data_eval_n_frames_input-1):
                            encoder_output, encoder_hidden, _,_,_  = model(inputs[:,ei,:,:,:], (ei==0))
                        decoder_input = inputs[:,-1,:,:,:] 
                        preds = []
                        for di in range(args.data_eval_n_frames_output):
                            decoder_output, decoder_hidden, output_image,_,_ = model(decoder_input, False, False)
                            decoder_input = output_image
                            preds.append(output_image.unsqueeze(1))
                        preds = torch.cat(preds, dim=1) 
                        loss = loss_fn(preds, outputs)
                        loss_value += loss.item()
                        _loss_value = loss_value / _step
                        loss_tensor = torch.tensor(_loss_value).cuda(rank)
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                        avg_loss = loss_tensor.item() / world_size
                        if rank == 0:
                            current_lr = scheduler.get_last_lr()[0]
                            message = f"Eval step {_step} - Eval loss: {avg_loss:.6f}"
                            eval_dataloader.set_description(message)
                            log_message(message)
                model.train()

            if rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                message = f"Epoch {ep+1}/{args.epochs} - Step {step} - Loss: {avg_loss:.6f} - LR: {current_lr:.6e}"
                train_dataloader.set_description(message)
                log_message(message)

            if rank == 0 and step % args.ckpt_step == 0:
                save_ckpt(model, ep, step, loss, args)

    cleanup_ddp()

if __name__ == "__main__":
    args = Args()
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.multiprocessing.spawn(run_ddp,
                                    args=(world_size, args,),
                                    nprocs=world_size,
                                    join=True)
    else:
        print("This script requires multiple GPUs to run.")

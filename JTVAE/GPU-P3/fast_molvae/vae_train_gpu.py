import torch
from functools import reduce
import operator as op
#print("Before imports: ")
#print(torch.cuda.mem_get_info()[0]/1000000000)
#print()

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torch.nn.parallel import DistributedDataParallel as DDP #Testing
from torch.distributed import init_process_group, destroy_process_group

import math, random, sys
import numpy as np
import argparse
from collections import deque
import pickle
import rdkit
import time
import gc, os

from fast_jtnn import *

import traceback

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_epoch', type=int, default=0)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=128)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

#JI - Modify default lr = 0.0007
parser.add_argument('--lr', type=float, default=0.0007)
#JI - Add argument min_lr to keep learning rate from dropping below this value - default = 0
parser.add_argument('--min_lr', type=float, default=0.0)

parser.add_argument('--clip_norm', type=float, default=50.0)

#JI - Modify beta default = 0.006 (constant) for after warmup, = 0 before (below in code)
parser.add_argument('--beta', type=float, default=0.006)

#JI - Change step_beta to 0.005, so it matches Original-JTVAE - next two parameters not used in new Training
parser.add_argument('--step_beta', type=float, default=0.005)
parser.add_argument('--max_beta', type=float, default=1.0)

#JI - Modify warmup to use 'epoch' unit, set default to 3
parser.add_argument('--warmup', type=int, default=3)

#JI -Change default number of epochs to 40
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--anneal_rate', type=float, default=0.9)

#JI - Modify anneal_iter default = 15000 (applied after warmup) - this parameter not used in new Training
parser.add_argument('--anneal_iter', type=int, default=15000)

#JI - kl_anneal_iter, save_iter not used in new Training - we save model after each epoch
parser.add_argument('--kl_anneal_iter', type=int, default=2000)
parser.add_argument('--print_iter', type=int, default=1000)
parser.add_argument('--save_iter', type=int, default=5000)

#SB - local_rank is a parameter used by DDP for distributed GPU training
#parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--mult_gpus", default=False, action='store_true')

#JI-Add - Default number of workers = 4
parser.add_argument('--num_workers', type=int, default=4)

args = parser.parse_args()
print (args)


vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

### STB - DDP related code
if args.mult_gpus == True:

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    init_process_group(backend='nccl')
###

model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda(device=local_rank)
if args.mult_gpus == True:
    #model = DDP(model, device_ids=[local_rank], output_device=local_rank) #Test DDP
    model = DDP(model, device_ids=[local_rank]) #Test DDP
print (model)

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

if args.load_epoch > 0:
    model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch)))

print ("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = args.load_epoch
beta0 = 0.0
beta  = args.beta
meters = np.zeros(4)

start_time = time.time()
curr_time = time.time()
bl_time = 0
p1_time = 0
failed = []

for epoch in range(args.epoch):
    shuffle = True
    loader = MolTreeFolder(args.train, vocab, args.batch_size, args.num_workers, shuffle, args.mult_gpus)
    
    for batch in loader:

        bl_time = bl_time + time.time() - curr_time
        curr_time = time.time()

        total_step += 1

        try:
            model.zero_grad()
            if epoch < args.warmup:
               loss, kl_div, wacc, tacc, sacc = model(batch, beta0)

            else:
               loss, kl_div, wacc, tacc, sacc = model(batch, beta)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

        except Exception as e:
            print("STB train EXCEPTION") #STB: TEMP
            print (e)

            #STB: If we want to print out the explicit exception with full trace
            # TODO: we could add an --explicit parameter to make this debugging simpler or a --debug mode
            #print(''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))

            continue
        

        p1_time = p1_time + time.time() - curr_time
        curr_time = time.time()

        meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            print ("[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model)))
            sys.stdout.flush()
            meters *= 0

            time_100 = time.time() - start_time
            print ('Total time = %.0f seconds' % time_100)
            print ('Batch processing time, model training time = %.0f, %.0f seconds' % (bl_time, p1_time))

#JI        if total_step % args.save_iter == 0:
#JI            torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(total_step))

#JI     if total_step % args.anneal_iter == 0:
#JI     if total_step % args.anneal_iter == 0 and epoch >= args.warmup:
#JI         scheduler.step()
#JI         print "learning rate: %.6f" % scheduler.get_lr()[0]

#JI     if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
#JI         beta = min(args.max_beta, beta + args.step_beta)

        curr_time = time.time()
        

    cur_lr = scheduler.get_last_lr()[0]
    print ("Current learning rate: %.6f" % cur_lr)
    new_lr = cur_lr*args.anneal_rate
    if new_lr > args.min_lr:
       scheduler.step()
       print ("New learning rate: %.6f" % new_lr)
    
    print ('Total epochs, iterations = %d, %d ' % (epoch, total_step))
    
    if local_rank == 0:
        if args.mult_gpus == True:
            ckp = model.module.state_dict()
        else:
            ckp = model.state_dict()
        torch.save(ckp, args.save_dir + "/model.epoch-" + str(epoch))
    
end_time = time.time()
tot_time = end_time - start_time
print ('Total time to run = %.0f seconds' % tot_time)

torch.distributed.destroy_process_group()
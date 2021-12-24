"""IIA-TCL training"""


from datetime import datetime
import os.path
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from itcl import itcl
from subfunc.showdata import *


# =============================================================
# =============================================================
def train(data,
          label,
          list_hidden_nodes,
          list_hidden_nodes_z,
          num_segment,
          initial_learning_rate,
          momentum,
          max_steps,
          decay_steps,
          decay_factor,
          batch_size,
          train_dir = None,
          weight_decay = 0,
          moving_average_decay = 0.999,
          summary_steps = 500,
          checkpoint_steps = 10000,
          save_file = 'model.pt',
          load_file = None,
          random_seed = None):
    """Build and train a model
    Args:
        data: data. 2D ndarray [num_data, num_comp]
        label: labels. 1D ndarray [num_data]
        list_hidden_nodes: number of nodes for each layer. 1D array [num_layer]
        list_hidden_nodes_z: number of nodes for each layer of MLP-z. 1D array [num_layer]
        num_segment: number of segment
        initial_learning_rate: initial learning rate
        momentum: momentum parameter (tf.train.MomentumOptimizer)
        max_steps: number of iterations (mini-batches)
        decay_steps: decay steps (tf.train.exponential_decay)
        decay_factor: decay factor (tf.train.exponential_decay)
        batch_size: mini-batch size
        train_dir: save directory
        weight_decay: weight decay
        moving_average_decay: (option) moving average decay of variables to be saved
        summary_steps: (option) interval to save summary
        checkpoint_steps: (option) interval to save checkpoint
        save_file: (option) name of model file to save
        load_file: (option) name of model file to load
        random_seed: (option) random seed
    Returns:
    """

    # Set random_seed
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define network
    model = itcl.Net([data.shape[1]]+list_hidden_nodes,
                      h_sizes_z=[int(data.shape[1]/2)]+list_hidden_nodes_z if list_hidden_nodes_z is not None else None,
                      num_class=num_segment)
    model = model.to(device)
    model.train()

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_factor)
    if train_dir is not None:
        writer = SummaryWriter(log_dir=train_dir)

    state_dict_ema = model.state_dict()

    trained_step = 0
    if load_file is not None:
        print("Load trainable parameters from {0:s}...".format(load_file))
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trained_step = checkpoint['step']

    # reshape data for fast batching (assume that the number of data in segments are somehow balanced)
    print("Reshaping data...")
    label_unique = np.unique(label).astype(np.int)
    num_segmentdata = np.zeros([num_segment], dtype=np.int)
    for ln in label_unique:
        num_segmentdata[ln] = np.sum(label==ln)
    x = np.zeros([num_segment, np.max(num_segmentdata).astype(np.int), data.shape[1]])
    for i, ln in enumerate(label_unique):
        x[i,0:num_segmentdata[i],:] = data[label==ln,:]
    # x: [segment, time, dim]

    # training iteration
    for step in range(trained_step, max_steps):
        start_time = time.time()
        scheduler.step()

        # Make shuffled batch
        x_batch = np.zeros([batch_size, x.shape[2]])
        y_batch = np.zeros([batch_size])
        for bn in range(batch_size):
            seg_idx = np.random.randint(0, num_segment)
            t_idx = np.random.randint(0, num_segmentdata[seg_idx])
            x_batch[bn, :] = x[seg_idx, t_idx, :]
            y_batch[bn] = seg_idx

        x_torch = torch.from_numpy(x_batch.astype(np.float32)).to(device)
        y_torch = torch.from_numpy(y_batch).type(torch.LongTensor).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits, h, hz = model(x_torch)
        loss = criterion(logits, y_torch)
        loss.backward()
        optimizer.step()

        # moving average of parameters
        state_dict_n = model.state_dict()
        for key in state_dict_ema:
            state_dict_ema[key] = moving_average_decay * state_dict_ema[key] \
                                  + (1.0 - moving_average_decay) * state_dict_n[key]

        # accuracy
        _, predicted = torch.max(logits.data, 1)
        accu_val = (predicted == y_torch).sum().item()/batch_size
        loss_val = loss.item()
        lr = scheduler.get_lr()[0]

        duration = time.time() - start_time

        assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

        # display stats
        if step % 100 == 0:
            num_examples_per_step = batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ('%s: step %d, lr = %f, loss = %.2f, accuracy = %3.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), step, lr, loss_val, accu_val * 100,
                                examples_per_sec, sec_per_batch))

        # save summary
        if step % summary_steps == 0 and train_dir is not None:
            writer.add_scalar('scalar/lr', lr, step)
            writer.add_scalar('scalar/loss', loss_val, step)
            writer.add_scalar('scalar/accu', accu_val, step)
            h_val = h.cpu().detach().numpy()
            h_comp = np.split(h_val, indices_or_sections=h.shape[1], axis=1)
            for (i, cm) in enumerate(h_comp):
                writer.add_histogram('h/h{0:d}'.format(i), cm)
            for k, v in state_dict_n.items():
                writer.add_histogram('w/{0:s}'.format(k), v)

        # Save the model checkpoint periodically.
        if step % checkpoint_steps == 0 and train_dir is not None:
            checkpoint_path = os.path.join(train_dir, save_file)
            torch.save({'step': step,
                        'model_state_dict':model.state_dict(),
                        'ema_state_dict': state_dict_ema,
                        'optimizer_state_dict':optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict()}, checkpoint_path)

    # evaluate by the whole data
    yall = np.tile(np.arange(x.shape[0]).reshape([-1,1]), [1, x.shape[1]])
    x_torch = torch.from_numpy(x.reshape([-1, x.shape[-1]]).astype(np.float32)).to(device)
    y_torch = torch.from_numpy(yall.reshape(-1)).type(torch.LongTensor).to(device)
    logits, h, hz = model(x_torch)
    loss = criterion(logits, y_torch)
    _, predicted = torch.max(logits.data, 1)
    accu_val = (predicted == y_torch).sum().item() / len(y_torch)
    loss_val = loss.item()

    # Save trained model ----------------------------------
    if train_dir is not None:
        save_path = os.path.join(train_dir, save_file)
        print("Save model in file: {0:s}".format(save_path))
        torch.save({'step': max_steps,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': state_dict_ema,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()}, save_path)

    return model.state_dict(), state_dict_ema, loss_val, accu_val

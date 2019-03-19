from imp import load_source
import numpy as np
import argparse
import sys
import time
import os

import torchvision
from torchvision import transforms
import torch
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint_sequential
    
means = np.array([0.485, 0.456, 0.406])
stds = np.array([0.229, 0.224, 0.225])

random_seed = 42
torch.manual_seed(random_seed)

def count_score(model, batch_gen, accuracy_list, gpu=False):
    model.train(False) # disable dropout / use averages for batch_norm
    for X_batch, y_batch in batch_gen:
        if gpu:
            logits = model(Variable(torch.FloatTensor(X_batch)).cuda())
        else:
            logits = model(Variable(torch.FloatTensor(X_batch)).cpu())

        y_pred = logits.max(1)[1].data
        accuracy_list.append(np.mean( (y_batch.cpu() == y_pred.cpu()).numpy() ))
    return accuracy_list

def train(model, opt, loss_fn, model_checkpoint_path, data_path, use_checkpoint=False, gpu=False, batch_size=128, epochs=100, effective_batch_size=None):
    transform = {
        'train': transforms.Compose([
            transforms.RandomRotation((-30,30)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])
    }
    
    dataset = torchvision.datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform['train'])
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [80000, 20000])


    train_batch_gen = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)
    val_batch_gen = torch.utils.data.DataLoader(val_dataset, 
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2)
    
    train_loss = []
    val_accuracy = []
    
    prev_val_accuracy = count_score(model, batch_gen=val_batch_gen, accuracy_list=[], gpu=gpu)
    prev_val_acc =  np.mean(prev_val_accuracy[-len(val_dataset) // batch_size :]) * 100
    
    if effective_batch_size is None:
        batches_per_update = 1
    else:
        batches_per_update = effective_batch_size / batch_size
    
    if use_checkpoint:
        segments = 4
        modules = [module for k, module in model._modules.items()]
        
    for epoch in range(epochs):
        # In each epoch, we do a full pass over the training data:
        start_time = time.time()
        model.train(True) # enable dropout / batch_norm training behavior
        forward_time = 0
        backward_time = 0

        for batch_i, (X_batch, y_batch) in enumerate(train_batch_gen):
            # train on batch
            start_counter = time.perf_counter()
            if gpu:
                X_batch = Variable(torch.FloatTensor(X_batch)).cuda()
                y_batch = Variable(torch.LongTensor(y_batch)).cuda()
                if use_checkpoint:
                    logits = checkpoint_sequential(modules, segments, X_batch)
                else:
                    logits = model.cuda()(X_batch)
            else:
                X_batch = Variable(torch.FloatTensor(X_batch)).cpu()
                y_batch = Variable(torch.LongTensor(y_batch)).cpu()
                if use_checkpoint:
                    logits = checkpoint_sequential(modules, segments, X_batch)
                else:
                    logits = model.cpu()(X_batch)
            loss = loss_fn(logits, y_batch)
            
            apply_counter = time.perf_counter()
            forward_time += apply_counter - start_counter
            
            loss.backward()
            backward_time += time.perf_counter() - apply_counter
            
            if (batch_i + 1) % batches_per_update == 0:
                opt.step()
                opt.zero_grad()

            train_loss.append(loss.data.cpu().numpy())
        
        val_accuracy = count_score(model, batch_gen=val_batch_gen, accuracy_list=val_accuracy, gpu=gpu)
        vall_acc =  np.mean(val_accuracy[-len(val_dataset) // batch_size :]) * 100

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(
            np.mean(train_loss[-len(train_dataset) // batch_size :])))
        print("  validation accuracy: \t\t\t{:.2f} %".format(vall_acc))
        torch.save(model.state_dict(), os.path.join(model_checkpoint_path, "model_{}_{:.2f}.pcl".format(epoch, vall_acc)))
        
        print("\t\tForward pass took  {:.3f} seconds".format(forward_time))
        print("\t\tBackward pass took {:.3f} seconds".format(backward_time))

        if vall_acc > prev_val_acc:
            prev_val_acc = vall_acc
            print("Saving new best model!")
            torch.save(model.state_dict(), os.path.join(model_checkpoint_path, "model_best.pcl"))

    return model

def validate(model, data_path, batch_size):
    transform = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])
    }
    
    test_dataset = torchvision.datasets.ImageFolder(os.path.join(data_path, 'new_val'), transform=transform['test'])
    test_batch_gen = torch.utils.data.DataLoader(test_dataset, 
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
    
    model.train(False) # disable dropout / use averages for batch_norm
    test_acc = []

    for X_batch, y_batch in test_batch_gen:
        logits = model(Variable(torch.FloatTensor(X_batch)).cuda())
        y_pred = logits.max(1)[1].data
        test_acc += list((y_batch.cpu() == y_pred.cpu()).numpy())
    
    test_accuracy = np.mean(test_acc)
    
    print("Final results:")
    print("  test accuracy:\t\t{:.2f} %".format(
        test_accuracy * 100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_enabled', default=False, action='store_true')
    parser.add_argument('--checkpoint', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model_module_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_checkpoint_path', type=str)
    parser.add_argument('--effective_batch_size', type=int)

    args = parser.parse_args()
    
    if args.gpu_enabled:
        torch.cuda.manual_seed(random_seed)

    load_source("model", args.model_module_path) 
    from model import get_model
    model, opt, loss_fn = get_model(model_path=args.model_path, gpu=args.gpu_enabled)
    model = train(model, opt, loss_fn, model_checkpoint_path=args.model_checkpoint_path, data_path=args.data_path, gpu=args.gpu_enabled, batch_size=args.batch_size, epochs=args.epochs, use_checkpoint=args.checkpoint, effective_batch_size=args.effective_batch_size)
    validate(model, data_path=args.data_path, batch_size=args.batch_size)
    
    print("Peak memory usage by Pytorch tensors: {:.2f} Mb".format((torch.cuda.max_memory_allocated() / 1024 / 1024)))
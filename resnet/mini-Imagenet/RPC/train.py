
import argparse
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import torch.optim as optim
import torch.nn as nn
import torch
from skip import resnet101
from one_epoch import train_one_epoch, val
from my_data import MyDataSet


def train():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('using {} to train'.format(device))

    transform_train=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    transform_val=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    data_root = './pth/mini-imagenet/'
    json_path = './pth/mini-imagenet/classes_name.json'
    train_datasets = MyDataSet(root_dir=data_root,
                               csv_name="this_train.csv",
                               json_path=json_path,
                               transform=transform_train,
                               )

    val_datasets   = MyDataSet(root_dir=data_root,
                               csv_name="this_val.csv",
                               json_path=json_path,
                               transform=transform_val,
                               )
    train_dataloader=DataLoader(train_datasets,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)
    val_dataloader=DataLoader(val_datasets,batch_size=128,shuffle=False,num_workers=2,pin_memory=True)
    


    net= resnet101(num_classes=100).to(device)

    optimizer=optim.SGD([p for p in net.parameters() if p.requires_grad],lr=0.1,momentum=0.9,weight_decay=5e-4,nesterov=True)
    loss=nn.CrossEntropyLoss()
    lr_schedule=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=500,min_lr=1e-3)


    writer = SummaryWriter("logs")

    for epoch in range(300):

        mean_loss=train_one_epoch(net,optimizer,loss,lr_schedule,epoch,train_dataloader,device,25,128)
        writer.add_scalar('train_loss',mean_loss,epoch)


        val_accuracy=val(val_dataloader,net,device,epoch)
        writer.add_scalar('val_acc',val_accuracy,epoch)
        if (epoch + 1) == 50:
            optimizer=optim.SGD([p for p in net.parameters() if p.requires_grad],lr=0.02,momentum=0.9,weight_decay=5e-4,nesterov=True)
        if (epoch + 1) == 100:
            optimizer=optim.SGD([p for p in net.parameters() if p.requires_grad],lr=0.004,momentum=0.9,weight_decay=5e-4,nesterov=True)
        if (epoch + 1) == 140:
            optimizer=optim.SGD([p for p in net.parameters() if p.requires_grad],lr=0.0008,momentum=0.9,weight_decay=5e-4,nesterov=True)

        if (epoch + 1) == 170:
            optimizer=optim.SGD([p for p in net.parameters() if p.requires_grad],lr=0.000001,momentum=0.9,weight_decay=5e-4,nesterov=True)




       
        torch.save(net, './w/' + str(epoch) + '_RPC.pth')

if __name__ == '__main__':
    train()

import time
import torch
 
def train_one_epoch(model,optimizer,loss,lr_schedule,epoch,dataloader,device,printf,batch):
    start=time.time()
    all_loss=0
    all_accNum=0
    model.train()
    for idx,(img,labels) in enumerate(dataloader):
        img=img.to(device)
        labels=labels.to(device)
        out=model(img)
        los=loss(out,labels)
 
        optimizer.zero_grad()
        los.backward()
        optimizer.step()
 
        all_loss+=los.item()
        cur_acc=(out.data.max(dim=1)[1]==labels).sum()
        all_accNum+=cur_acc

        if (idx%printf)==0:
            print('about train:-------> epoch:{} training:[{}/{}] loss:{:.6f} accuracy:{:.6f}% lr:{}'.format(epoch,idx,len(dataloader),los.item(),cur_acc*100/len(labels),optimizer.param_groups[0]['lr']))
 

 
    end=time.time()

    all_loss/=len(dataloader)
    acc=all_accNum*100/(len(dataloader)*batch)
    print('about train:-------> epoch:{} time:{:.2f} seconds training_loss:{:.6f} training_accuracy:{:.6f}%'.format(epoch,end-start,all_loss,acc))
    return all_loss
 
@torch.no_grad()
def val(dataloader,model,device,epoch):
    start=time.time()
    model.eval()
    all_acc=0
    for idx,(img,labels) in enumerate(dataloader):
        img=img.to(device)
        labels=labels.to(device)
        out=model(img)
 
        cur_accNum=(out.data.max(dim=1)[1]==labels).sum()/len(labels)
        all_acc+=cur_accNum
    end=time.time()
    print('about val:-------> epoch:{} val_time:{:.2f} seconds val_accuracy:{:.6f}%'.format(epoch,end-start,all_acc*100/len(dataloader)))
    return all_acc/len(dataloader)
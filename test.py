<<<<<<< HEAD
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.frcnn import FasterRCNN
from trainer import FasterRCNNTrainer
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
def fit_ont_epoch(net,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda,train_util):
    """

    :param net:                 模型
    :param epoch:               第epoch世代
    :param epoch_size:          一个世代训练集的batch总数
    :param epoch_size_val:      一个世代验证集需进行的batch总数
    :param gen:                 训练样本dataloader
    :param genval:              测试样本dataloader
    :param Epoch:               需要迭代的总次数
    :param cuda:                是否使用cuda
    :param train_util:          所有的 loss
    :return:
    """
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    val_toal_loss = 0
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            imgs, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor)).cuda()
                    """  加不加这个Variable效果是一样的
                    在旧版本中variable和tensor的区别在于，variable可以进行误差的反向传播，而tensor不可以。
                    但是现在在新版本的torch中可以直接使用tensor而不需要使用variable。
                    从0.4起, Variable 正式合并入Tensor, Variable 本来实现的自动微分功能，Tensor就能支持。
                    读者还是可以使用Variable(tensor), 但是这个操作其实什么都没做。建议读者以后直接使用tensor.
                    """
                else:
                    imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))

            losses = train_util.train_step(imgs, boxes, labels, 1)  #返回训练误差

            rpn_loc, rpn_cls, roi_loc, roi_cls, total = losses
            total_loss += total.item()
            rpn_loc_loss += rpn_loc.item()
            rpn_cls_loss += rpn_cls.item()
            roi_loc_loss += roi_loc.item()
            roi_cls_loss += roi_cls.item()
            
            pbar.set_postfix(**{'total'    : total_loss / (iteration + 1), 
                                'rpn_loc'  : rpn_loc_loss / (iteration + 1),  
                                'rpn_cls'  : rpn_cls_loss / (iteration + 1), 
                                'roi_loc'  : roi_loc_loss / (iteration + 1), 
                                'roi_cls'  : roi_cls_loss / (iteration + 1), 
                                'lr'       : get_lr(optimizer)})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs,boxes,labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor)).cuda()
                else:
                    imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))

                train_util.optimizer.zero_grad()
                losses = train_util.forward(imgs, boxes, labels, 1)
                _, _, _, _, val_total = losses

                val_toal_loss += val_total.item()

            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))
    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))
    
if __name__ == "__main__":
    # torch.cuda.empty_cache()   #清除GPU缓存
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda = True
    #----------------------------------------------------#
    #   训练之前一定要修改NUM_CLASSES
    #   修改成所需要区分的类的个数。
    #----------------------------------------------------#
    NUM_CLASSES = 6
    #-------------------------------------------------------------------------------------#
    #   input_shape是输入图片的大小，默认为800,800,3，随着输入图片的增大，占用显存会增大
    #-------------------------------------------------------------------------------------#
    input_shape = [600,600,3]
    #----------------------------------------------------#
    #   使用到的主干特征提取网络
    #   vgg或者resnet50
    #----------------------------------------------------#
    backbone = "resnet50"
    model = FasterRCNN(NUM_CLASSES,backbone=backbone)

    # #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #------------------------------------------------------#
    model_path = 'model_data/voc_weights_resnet.pth'
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()    #OrderedDict :328
    pretrained_dict = torch.load(model_path, map_location=device) #OrderedDict :328
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)   #更新模型的预训练参数
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()   #设置model为训练模式
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    annotation_path = 'train.txt'
    #----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        lr = 1e-4
        Batch_size = 2
        Init_Epoch = 0
        Freeze_Epoch = 50
        
        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)  #调整学习率机制

        train_dataset = FRCNNDataset(lines[:num_train], (input_shape[0], input_shape[1]), is_train=True)   #进行数据增强
        val_dataset   = FRCNNDataset(lines[num_train:], (input_shape[0], input_shape[1]), is_train=False)  #不进行数据增强   图像等比例缩放,无图像区域用128填充
        gen     = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=0, pin_memory=True,
                                drop_last=True, collate_fn=frcnn_dataset_collate)
        # pin_memory:会自动将获取的数据张量放入固定的内存中，从而更快地将数据传输到支持CUDA的GPU。
        # collate_fn:用于合并样本列表以形成小批量的Tensor ,批量加载数据时使用
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=0, pin_memory=True,
                                drop_last=True, collate_fn=frcnn_dataset_collate)
                        
        epoch_size = num_train // Batch_size   #每一个世代训练需要总batch数
        epoch_size_val = num_val // Batch_size #每一个世代验证所需要的总batch数
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.extractor.parameters():  #冻结特征提取部分
            param.requires_grad = False

        # ------------------------------------#
        #   冻结bn层
        # ------------------------------------#
        model.freeze_bn()

        train_util = FasterRCNNTrainer(model, optimizer)

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_ont_epoch(net,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda,train_util)
            lr_scheduler.step()

    if True:
        lr = 1e-5
        Batch_size = 2
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        train_dataset = FRCNNDataset(lines[:num_train], (input_shape[0], input_shape[1]), is_train=True)
        val_dataset   = FRCNNDataset(lines[num_train:], (input_shape[0], input_shape[1]), is_train=False)
        gen     = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=frcnn_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=frcnn_dataset_collate)
                        
        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.extractor.parameters():
            param.requires_grad = True

        # ------------------------------------#
        #   冻结bn层
        # ------------------------------------#
        model.freeze_bn()

        train_util = FasterRCNNTrainer(model,optimizer)

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_ont_epoch(net,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda,train_util)
            lr_scheduler.step()
=======
version https://git-lfs.github.com/spec/v1
oid sha256:1bc8a2f33015725e98d78920af8e4c5ace6c94ee5ac1ab86c79643b15e63a009
size 11460
>>>>>>> 9e98159e6ada6bd9974b33f8d00d99a56d693a76

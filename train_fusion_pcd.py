import os
import math
import torch
import argparse
import numpy as np
import torch.optim as optim
from datetime import datetime
from torchvision import transforms
from HAR_dataset import HARDataSet
from models.create_model import create_model
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from utils import evaluate_fusion_range_doppler
from utils import train_one_epoch_pcd, evaluate_pcd
from utils import train_one_epoch_fusion_range_doppler
from utils import train_one_epoch_fusion_pcd, evaluate_fusion_pcd
from utils import read_split_data, train_one_epoch, evaluate, check_path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)


    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = HARDataSet(signals_path=train_images_path,
                              signals_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = HARDataSet(signals_path=val_images_path,
                            signals_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn,
                                             drop_last=True)

    model = create_model(model_name=args.model_arch, num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    now = datetime.now()
    save_dir = 'Demo_{}'.format(now.strftime('%Y-%m-%d_%H:%M:%S'))
    logfile_save_path = os.path.join('./logs', args.model_arch, save_dir)
    weights_save_path = os.path.join('./weights', args.model_arch, save_dir)
    check_path(logfile_save_path)
    check_path(weights_save_path)

    # 创建列表保存损失和准确率
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(args.epochs):
        # train
        # fusion_mode为0时，此时输入数据只有time-doppler; 模型为mobile_vit
        if args.fusion_mode == 0:
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch)

            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)

        # fusion_mode为1时，此时输入数据为time_doppler，time_range; 模型为双分支mobile_vit
        if args.fusion_mode == 1:
            train_loss, train_acc = train_one_epoch_fusion_range_doppler(model=model,
                                                            optimizer=optimizer,
                                                            data_loader=train_loader,
                                                            device=device,
                                                            epoch=epoch)
            scheduler.step()


            val_loss, val_acc = evaluate_fusion_range_doppler(model=model,
                                                 data_loader=val_loader,
                                                 device=device,
                                                 epoch=epoch)

        # fusion_mode为1时，此时输入数据为time_range，time_range; 模型为双分支mobile_vit
        if args.fusion_mode == 2:
            train_loss, train_acc = train_one_epoch_fusion_pcd(model=model,
                                                               optimizer=optimizer,
                                                               data_loader=train_loader,
                                                               device=device,
                                                               epoch=epoch)

            val_loss, val_acc = evaluate_fusion_pcd(model=model,
                                                    data_loader=val_loader,
                                                    device=device,
                                                    epoch=epoch)

        if args.fusion_mode == 3:
            train_loss, train_acc = train_one_epoch_pcd(model=model,
                                                        optimizer=optimizer,
                                                        data_loader=train_loader,
                                                        device=device,
                                                        epoch=epoch)

            val_loss, val_acc = evaluate_pcd(model=model,
                                             data_loader=val_loader,
                                             device=device,
                                             epoch=epoch)


            scheduler.step()



        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        train_loss_list.append(round(train_loss, 3))
        train_acc_list.append(round(train_acc, 3))
        val_loss_list.append(round(val_loss, 3))
        val_acc_list.append(round(val_acc, 3))
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), weights_save_path + "/model-{}.pth".format(epoch))

    # 使用字典的形式保存log文件
    log_dict = {'train_loss': train_loss_list, 'train_acc': train_acc_list,
                'val_loss': val_loss_list, 'val_acc': val_acc_list}

    np.save(os.path.join(logfile_save_path, 'log.npy'), log_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--model_arch', type=str, default='diharnet')
    parser.add_argument('--fusion_mode', type=int, default=3,
                        help='0-> only time-doppler; '
                             '1-> time-doppler + time-range;'
                             '2-> time-doppler + pcd;'
                             '3-> only pcd;')
    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data_path', type=str, default="./create_dataset/parsed_data_clip_1/train/time_doppler/")
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:3', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

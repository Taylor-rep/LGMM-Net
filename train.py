import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from loader import *
from models.LGMM_net import LGMM_net
from models.UNet import MALUNet,U_Net
from engine import *
import os
import sys
from utils import *
from configs.config_setting import setting_config
import warnings
warnings.filterwarnings("ignore")
from utils import BceDiceLoss

def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = getattr(config, 'resume_model', './pretrained_weights/best.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)
    global logger
    logger = get_logger('train', log_dir)
    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    if getattr(config, 'semi_supervised', False):
        print('#----------半监督Mean-Teacher模式----------#')
        labeled_dataset = isic_loader(path_Data = config.data_path, train = True)
        labeled_loader = DataLoader(labeled_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)
        unlabeled_dataset = UnlabeledISICLoader(path_Data = config.data_path)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)
        val_dataset = isic_loader(path_Data = config.data_path, train = False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)
        test_dataset = isic_loader(path_Data = config.data_path, train = False, Test = True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)
    else:
        print('#----------全监督标准模式----------#')
        train_dataset = isic_loader(path_Data = config.data_path, train = True)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)
        val_dataset = isic_loader(path_Data = config.data_path, train = False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)
        test_dataset = isic_loader(path_Data = config.data_path, train = False, Test = True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)

    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    if getattr(config, 'semi_supervised', False):
        student_model = LGMM_net(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            c_list=model_cfg['c_list'],
            split_att=model_cfg['split_att'],
            bridge=model_cfg['bridge']
        )
        teacher_model = LGMM_net(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            c_list=model_cfg['c_list'],
            split_att=model_cfg['split_att'],
            bridge=model_cfg['bridge']
        )
        teacher_model.load_state_dict(student_model.state_dict())
        for p in teacher_model.parameters():
            p.requires_grad = False
        gpu_ids = [0]
        student_model = torch.nn.DataParallel(student_model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
        teacher_model = torch.nn.DataParallel(teacher_model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
    else:
        model = LGMM_net(num_classes=model_cfg['num_classes'],
                                   input_channels=model_cfg['input_channels'],
                                   c_list=model_cfg['c_list'],
                                   split_att=model_cfg['split_att'],
                                   bridge=model_cfg['bridge'])
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0], output_device=0)

        print(f"Running model: {model.module.__class__.__name__}")
        cal_params_flops(model.module, size=256, logger=logger)
        # 重新放到GPU和DataParallel，确保参数都在cuda:0
        model = model.module.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0], output_device=0)


    print('#----------Prepareing loss, opt, sch and amp----------#')
    if config.criterion == 'BceDiceLoss':
        criterion = BceDiceLoss()
    elif config.criterion == 'DiceLoss':
        criterion = DiceLoss()
    elif config.criterion == 'BCELoss':
        criterion = BCELoss()
    elif config.criterion == 'CrossEntropyLoss':
        criterion = CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {config.criterion}")
    if getattr(config, 'semi_supervised', False):
        optimizer = get_optimizer(config, student_model)
        scheduler = get_scheduler(config, optimizer)
    else:
        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    # if os.path.exists(resume_model):
    #     print('#----------Resume Model and Other params----------#')
    #     checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
    #     if 'model_state_dict' in checkpoint:
    #         state_dict = checkpoint['model_state_dict']
    #     else:
    #         state_dict = checkpoint
    #     if getattr(config, 'semi_supervised', False):
    #         student_model.module.load_state_dict(state_dict, strict=False)
    #     else:
    #         model.module.load_state_dict(state_dict, strict=False)
    #     print(f'Loaded weights from {resume_model} (strict=False), optimizer/scheduler NOT loaded.')

    print('#----------Training----------#')
    if getattr(config, 'semi_supervised', False):
        lambda_consistency = config.lambda_consistency
        ema_alpha = config.ema_alpha
        unlabeled_iter = iter(unlabeled_loader)
        for epoch in range(start_epoch, config.epochs + 1):
            student_model.train()
            teacher_model.eval()
            for i, (img_l, mask_l) in enumerate(labeled_loader):
                try:
                    img_u = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    img_u = next(unlabeled_iter)
                img_l, mask_l = img_l.cuda(), mask_l.cuda()
                img_u = img_u.cuda()
                optimizer.zero_grad()
                # 1. 有标签数据监督损失
                pred_l = student_model(img_l)
                loss_sup = criterion(pred_l, mask_l)
                # 2. 无标签一致性损失
                with torch.no_grad():
                    teacher_pred_u = teacher_model(img_u)
                student_pred_u = student_model(img_u)
                loss_consistency = torch.nn.functional.mse_loss(student_pred_u, teacher_pred_u)
                # 3. 总损失
                loss = loss_sup + lambda_consistency * loss_consistency
                loss.backward()
                optimizer.step()
                # 4. EMA更新教师模型
                with torch.no_grad():
                    for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
                        t_param.data = ema_alpha * t_param.data + (1 - ema_alpha) * s_param.data
            scheduler.step()
            loss = val_one_epoch(
                    val_loader,
                    student_model,
                    criterion,
                    epoch,
                    logger,
                    config
                )
            if loss < min_loss:
                torch.save(student_model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                min_loss = loss
                min_epoch = epoch
            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': student_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth')) 
        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            print('#----------Testing----------#')
            best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
            student_model.module.load_state_dict(best_weight)
            loss = test_one_epoch(
                    test_loader,
                    student_model,
                    criterion,
                    logger,
                    config,
                )
            os.rename(
                os.path.join(checkpoint_dir, 'best.pth'),
                os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
            )
    else:
        for epoch in range(start_epoch, config.epochs + 1):
            torch.cuda.empty_cache()
            train_one_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                logger,
                config,
                scaler=scaler
            )
            loss = val_one_epoch(
                    val_loader,
                    model,
                    criterion,
                    epoch,
                    logger,
                    config
                )
            if loss < min_loss:
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                min_loss = loss
                min_epoch = epoch
            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth')) 
        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            print('#----------Testing----------#')
            best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
            model.module.load_state_dict(best_weight)
            loss = test_one_epoch(
                    test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                )
            os.rename(
                os.path.join(checkpoint_dir, 'best.pth'),
                os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
            )

if __name__ == '__main__':
    config = setting_config()
    main(config)
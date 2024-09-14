# 以置信度排序进行过滤
from segment_anything import sam_model_registry
import torch.nn as nn
import torch
import argparse
import os
from utils import FocalDiceloss_IoULoss, generate_point, save_masks, save_masks2
from torch.utils.data import DataLoader
from DataLoader import TestingDataset
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import logging
import datetime
import cv2
import random
import csv
import json
from datetime import datetime
from quant_utils import build_model, adapter_channel_wise
from quant import *
from utils import *
from generate_data_TU import generate_data2, prompt_and_decoder_for_distill
import copy
import matplotlib.pyplot as plt



from tensorboardX import SummaryWriter


class post_LN_activation:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.feature = output
    def remove(self):
        self.hook.remove()


def show_metrics(metrics_list, name):
    if len(metrics_list) != 0:
        iou_s,dice_s = 0, 0
        for iou, dice in metrics_list:
            iou_s+=iou
            dice_s+=dice
        iou, dice = iou_s/len(metrics_list), dice_s/len(metrics_list)
        print(name, f"Average metrics: iou:{iou:.4f} dice:{dice:.4f}")
        
def avg_metrics(metrics_list_list, name_list):
    if len(metrics_list_list) != 0:
        iou_sum, dice_sum = 0, 0
        for metrics_list, name in zip(metrics_list_list, name_list):
            if len(metrics_list)>0:
                iou_s,dice_s = 0, 0
                for iou, dice in metrics_list:
                    iou_s+=iou
                    dice_s+=dice
                iou, dice = iou_s/len(metrics_list), dice_s/len(metrics_list)
                iou_sum+=iou
                dice_sum+=dice
                print(name, f"Average metrics: iou:{iou:.4f} dice:{dice:.4f}")
        iou_avg ,dice_avg = iou_sum/len(metrics_list_list), dice_sum/len(metrics_list_list)
        
        print(name_list,'average metrics:','iou:'+str(iou_avg),'dice:'+str(dice_avg))
        
def test_all_dataset(args, logger, model):
    ct_metrics_list = []
    dmcp_metrics_list = []
    edcp_metrics_list = []
    fudus_metrics_list = []
    mr_metrics_list = []
    
    
    if args.img_channel==3:
        logger.info('****************testing SA-Med datasets-20: color--212***************')
        model_test(args, model, args.data_path_SA_Med20)    
    
        
        logger.info('****************testing SA-Med datasets-7: (dermoscopy) isic2016-2018[330]***************')
        dmcp_metrics_list.append(model_test(args, model, args.data_path_SA_Med7))
        logger.info('****************testing SA-Med datasets-8: (endoscopy) EndoVis2017[648]***************')
        edcp_metrics_list.append(model_test(args, model, args.data_path_SA_Med8))
        logger.info('****************testing SA-Med datasets-9: (endoscopy) EndoVis2015, cvc_clinicdb, kvasir[127]***************')
        edcp_metrics_list.append(model_test(args, model, args.data_path_SA_Med9))
        logger.info('****************testing SA-Med datasets-10: (fundus) PALM19, gamma[128]***************')
        fudus_metrics_list.append(model_test(args, model, args.data_path_SA_Med10))
    
        logger.info('****************testing Kvasir datasets***************')
        model_test(args, model, args.data_path_Kvasir)
    else:
        logger.info('****************testing SA-Med datasets-4: (CT) AbdomenCT1k[3361]***************')
        ct_metrics_list.append(model_test(args, model, args.data_path_SA_Med4))
        logger.info('****************testing SA-Med datasets-19: gray--651***************')
        model_test(args, model, args.data_path_SA_Med19)
        logger.info('****************testing SA-Med datasets-21: (MRI) MSD-heart-prostate[193]***************')
        mr_metrics_list.append(model_test(args, model, args.data_path_SA_Med21))
        logger.info('****************testing synapse***************')
        model_test(args, model, args.data_path_synapse)
       

    
    
    
def model_test(args, model, dataset_dir):
    criterion = FocalDiceloss_IoULoss()
    test_dataset = TestingDataset(data_path=dataset_dir, image_size=args.image_size, mode='test', requires_name=True, point_num=args.point_num, return_ori_mask=True, prompt_path=args.prompt_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print('Test data:', len(test_loader))
    save_folder = dataset_dir.split('/')[-1]
   
    test_pbar = tqdm(test_loader)
    l = len(test_loader)

    model.eval()
    test_loss = []
    test_iter_metrics = [0] * len(args.metrics)
    test_metrics = {}
    prompt_dict = {}

    for i, batched_input in enumerate(test_pbar):
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        ori_images = batched_input["ori_image"]
        original_size = batched_input["original_size"]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]
        if args.prompt_path is None:
            prompt_dict[img_name] = {
                        "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                        "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                        "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
                        }
        
        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])

        
        if args.boxes_prompt:
            save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt",args.test_mode)
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
            points_show = None

        else:
            save_path = os.path.join(f"{args.work_dir}", args.run_name, f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt")
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]
    
            for iter in range(args.iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
                if iter != args.iter_point-1:
                    batched_input = generate_point(masks, labels, low_res_masks, batched_input, args.point_num)
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels, dim=1)
  
            points_show = (torch.concat(point_coords, dim=1), torch.concat(point_labels, dim=1))

        masks, pad = postprocess_masks2(low_res_masks, args.image_size, original_size)      # 与Dataloader的train_transforms2对应，统一使用缩放来处理输入图片，不用pad
        
        if args.save_pred:
            if i % 10 == 0:
                save_masks2(args,ori_images, ori_labels, masks, save_path, img_name, args.image_size, original_size, pad, batched_input.get("boxes", None), points_show, True, save_folder)

        loss = criterion(masks, ori_labels, iou_predictions)
        test_loss.append(loss.item())

        test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]

        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]
  
    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}

    average_loss = np.mean(test_loss)
    
    logger.info(f"Test loss: {average_loss:.4f}, metrics: {test_metrics}")
    return test_iter_metrics
    



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="/home/zhangjing/SAM-Med2D_q/save_result_vis", help="work dir")
    parser.add_argument("--run_name", type=str, default="sammed", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    
    parser.add_argument("--data_path_FLARE", type=str, default="/home/zhangjing/MedSAM/data/FLARE22Train_png2", help="train data path") 
    parser.add_argument("--data_path_Kvasir", type=str, default="/home/zhangjing/MedSAM/data/Kvasir-SEG", help="train data path") 
    parser.add_argument("--data_path_ATLAS", type=str, default="/home/zhangjing/MedSAM/data/ATLAS/png", help="train data path") 
    parser.add_argument("--data_path_ToothFairy", type=str, default="/home/zhangjing/MedSAM/data/ToothFairy/train", help="train data path") 
    parser.add_argument("--data_path_SA_Med1", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset1", help="train data path") 
    parser.add_argument("--data_path_SA_Med2", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset2", help="train data path") 
    parser.add_argument("--data_path_SA_Med3", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset3", help="train data path") 
    parser.add_argument("--data_path_SA_Med4", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset4", help="train data path") 
    parser.add_argument("--data_path_SA_Med5", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset5", help="train data path") 
    parser.add_argument("--data_path_SA_Med6", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset6", help="train data path") 
    parser.add_argument("--data_path_SA_Med7", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset7", help="train data path") 
    parser.add_argument("--data_path_SA_Med8", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset8", help="train data path") 
    parser.add_argument("--data_path_SA_Med9", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset9", help="train data path") 
    parser.add_argument("--data_path_SA_Med10", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset10", help="train data path") 
    parser.add_argument("--data_path_SA_Med11", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset11", help="train data path") 
    parser.add_argument("--data_path_SA_Med12", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset12", help="train data path") 
    parser.add_argument("--data_path_SA_Med13", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset13", help="train data path") 
    parser.add_argument("--data_path_SA_Med14", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset14", help="train data path") 
    parser.add_argument("--data_path_SA_Med15", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset15", help="train data path") 
    parser.add_argument("--data_path_SA_Med16", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset16", help="train data path") 
    parser.add_argument("--data_path_SA_Med17", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset17", help="train data path") 
    parser.add_argument("--data_path_SA_Med18", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset18", help="train data path") 
    parser.add_argument("--data_path_SA_Med19", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset19", help="train data path") 
    parser.add_argument("--data_path_SA_Med20", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset20", help="train data path") 
    parser.add_argument("--data_path_SA_Med21", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/SAMed_dataset21", help="train data path") 
    parser.add_argument("--data_path_synapse", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/synapse_png", help="train data path") 
    parser.add_argument("--data_path_actual_calib", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/synapse_png/imgs/case0038_sid025.png", help="train data path") 
    parser.add_argument("--calib_png_data_path", type=str, default="/home/zhangjing/MedSAM_quant/Med_datasets/synapse_png/imgs/case0038_sid025.png", help="train data path") 
    parser.add_argument("--SW_log_dir", type=str, default="./SW_log/", help="train data path") 
    parser.add_argument("--dataset", type=str, default="Kvasir", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--boxes_prompt", type=bool, default=True, help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--iter_point", type=int, default=1, help="iter num") 
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=False, help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--save_pred_path", type=str, default='./result_vis/', help="save reslut")
    parser.add_argument("--save_pred", type=bool, default=False, help="save reslut")
    parser.add_argument('--w_bits', default=4, type=int, help='bit-precision of weights')
    parser.add_argument('--a_bits', default=4, type=int, help='bit-precision of activation')
    parser.add_argument('--calib_batch_size', default=1, type=int, help='calib_batch_size')
    parser.add_argument('--PLNA_channel_wise', default=True, type=bool, help='post LN activation')
    parser.add_argument('--log2_quant', default=True, type=bool, help='post LN activation')
    parser.add_argument('--AQ', default=True, type=bool, help='post LN activation')
    parser.add_argument('--WQ', default=True, type=bool, help='post LN activation')
    parser.add_argument('--test_mode', default='generated_data_calib', type=str, help='noize_calib, train_data_calib_r, noize_calib, generated_data_calib, joke_calib, FLARE_calib_b1, load_data_calib, real_data_calib')
    parser.add_argument('--load_calib_data_path', default='calib_data/generated_data_2024_08_21_02_41.pth', type=str)
    parser.add_argument('--ZQ_lr', default=0.2, type=float, help='post LN activation')
    parser.add_argument('--distill_data_path', default='./calib_data/generated_data', type=str, help='post LN activation')
    parser.add_argument('--target_label', default='random_1', help='TU_Synapse, random_1',type=str)
    parser.add_argument('--show_time', default='',type=str)
    parser.add_argument('--img_channel', default=3,type=int)
    
    # TU_args
    parser.add_argument('--TU_vit_name', default='R50-ViT-B_16', type=str)
    parser.add_argument('--TU_num_classes', default=9, type=int)
    parser.add_argument('--TU_n_skip', default=3, type=int)
    parser.add_argument('--TU_vit_patches_size', default=16, type=int)
    parser.add_argument('--TU_img_size', default=224, type=int)
    parser.add_argument('--TU_ckpt_path', default='ckpt_TU/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224.pth', type=str)
    
    # 超参
    parser.add_argument('--para_class', default=0.5, type=float, help='gray:0.5, color:0.5')
    parser.add_argument('--para_seg', default=1, type=float, help='gray:1, color:1')
    parser.add_argument('--para_PSE', default=0.05, type=float, help='gray:0.05, color:0.1')
    parser.add_argument('--para_tv', default=0.01, type=float, help='gray:0.01, color:0.001')
    
    
    args = parser.parse_args()
    if args.iter_point > 1:
        args.point_num = 1
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input




def postprocess_masks2(low_res_masks, image_size, original_size):
    # 没有pad，无论图片大小都执行缩放
    ori_h, ori_w = original_size
    
    masks = F.interpolate(low_res_masks, original_size, mode="bilinear", align_corners=False)
    pad = None
    return masks, pad


def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = ddp_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )
    
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions


def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True

def calib_qmodel(args, q_model):
    q_model.eval()
    if args.test_mode == 'noize_calib':
        calib_gausiion_noize(args, q_model)
    elif args.test_mode == 'train_data_calib_r':
        calib_random_train_data2(args, q_model)
    elif args.test_mode == 'joke_calib':
        calib_joke_data(args, q_model)
    elif args.test_mode == 'FLARE_calib_b1':
        calib_FLARE_b1(args, q_model)
    elif args.test_mode == 'generated_data_calib':
        calib_distilled2(args, q_model)
    elif args.test_mode == 'load_data_calib':
        calib_load_data(args, q_model, args.load_calib_data_path)
    elif args.test_mode == 'real_data_calib':
        calib_single_png(args, q_model, args.calib_png_data_path)
    
    

def main(args):
    global logger
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    current_time = datetime.now()
    show_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    args.distill_data_path = args.distill_data_path+'_'+show_time+'.pth'
    args.show_time = show_time
    
    
    if args.img_channel==3:
        args.para_class=0.6
        args.para_seg=1
        args.para_PSE=0.01
        args.para_tv=0.001
            
    # 创建Logger对象
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    # 创建文件处理器
    file_handler = logging.FileHandler('./ab_log/'+'log_info_'+args.show_time+'.log')
    file_handler.setLevel(logging.DEBUG)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    

    print('*'*100)
    for key, value in vars(args).items():
        logger.info(key + ': ' + str(value))
    print('*'*100)
    writer = SummaryWriter(args.SW_log_dir)

    model = sam_model_registry[args.model_type](args).to(args.device)

    if args.test_mode=='FP':
        test_all_dataset(args, logger, model)
        return
    
    b_model = build_model(model)
  
    
    if args.test_mode == 'generated_data_calib':
        generate_data2(args, b_model, writer)
    
    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    print('Performing initial quantization ...')
    q_model = quant_image_encoder(args, b_model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.cuda()
    q_model.eval()
    
    # calibrating q_model 
    print('Performing calibration ...')
    set_quant_state(q_model, input_quant=args.AQ, weight_quant=args.WQ)
    calib_qmodel(args, q_model)
        
    # Scale reparameterization
    print('Performing scale reparameterization ...')
    scale_reparameter(args, q_model)
    
    # Re-calibration
    set_quant_state(q_model, input_quant=args.AQ, weight_quant=args.WQ)
    calib_qmodel(args, q_model)
        
    test_all_dataset(args, logger, q_model)
    
    
    print('*'*100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*'*100)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)

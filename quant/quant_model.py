import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader
from DataLoader import TestingDataset
from quant_utils import MatMul
import random

from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul
from test_quant import to_device
from utils import train_transforms, get_boxes_from_mask, init_point_sampling, train_transforms2
import cv2
from torch.nn import Parameter
from PIL import Image
import numpy as np

def calib_random_train_data(args, model):
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    test_dataset = TestingDataset(data_path=args.data_path, image_size=args.image_size, mode='test', requires_name=True, point_num=args.point_num, return_ori_mask=True, prompt_path=args.prompt_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
    for i, batched_input in enumerate(test_loader):
        if i > 0:
            return
        batched_input = to_device(batched_input, args.device)
        print('calib...')
        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])
        torch.save(batched_input["image"], 'calib_data/calib_data_random_train.pth')

def calib_random_train_data2(args, model):
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    test_dataset = TestingDataset(data_path=args.data_path_actual_calib, image_size=args.image_size, mode='test', requires_name=True, point_num=args.point_num, return_ori_mask=True, prompt_path=args.prompt_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
    img_batch_list = []
    sample_boxes_gts = {"boxes":[], "gts":[]}
    
    for i, batched_input in enumerate(test_loader):
        if i >= args.calib_batch_size:
            break
        # batched_input = to_device(batched_input, args.device)
        img_batch_list.append(batched_input["image"].float().cuda())
        sample_boxes_gts["boxes"].append(batched_input["boxes"])
        sample_boxes_gts["gts"].append(batched_input["label"])
    torch.save(sample_boxes_gts, 'sample_boxes_gts/FLARE_1.pth')
    print('saved')
    img_batch = torch.cat(img_batch_list, dim=0)
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    for id, img_tensor in enumerate(img_batch_list):
        img_np = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_np = img_np * pixel_std + pixel_mean
        print(img_np)
        image = Image.fromarray((img_np).astype(np.uint8))
        image.save('./calib_data/calib_train_data_'+str(id)+'.png')
    print('calib...')
    with torch.no_grad():
        image_embeddings = model.image_encoder(img_batch)
    torch.save(batched_input["image"], 'calib_data/calib_data_random_train.pth')

def calib_gausiion_noize(args, model):
    calib_data=torch.randn((args.calib_batch_size, 1, args.image_size, args.image_size)).cuda()
    print('calib...')
    with torch.no_grad():
        image_embeddings = model.image_encoder(calib_data)

def calib_gausiion_noize_3C(args, model):
    calib_data=torch.randn((args.calib_batch_size, 3, args.image_size, args.image_size)).cuda()
    print('calib...')
    with torch.no_grad():
        image_embeddings = model.image_encoder(calib_data)
        
def calib_train_data(args, model):
    calib_data = torch.load('calib_data/calib_data_random_train.pth')
    calib_data = calib_data.float().cuda()
    with torch.no_grad():
        image_embeddings = model.image_encoder(calib_data)
        
def calib_single_png(args, model, calib_png_data_path):
    image_array = cv2.imread(calib_png_data_path)
    print('calibrate png path:',calib_png_data_path)
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    image_array = (image_array - pixel_mean) / pixel_std
    
    h, w, _ = image_array.shape
    transforms = train_transforms(args.image_size, h, w)
    augments = transforms(image=image_array)
    image = augments['image'].float().unsqueeze(0).cuda()
    with torch.no_grad():
        image_embeddings = model.image_encoder(image)
        
        
def calib_FLARE_b1(args, model):
    calib_data = torch.load('calib_data/calib_data_FLARE.pth')
    with torch.no_grad():
        image_embeddings = model.image_encoder(calib_data)
        
def calib_joke_data(args, model):
    image_path = '/home/zhangjing/SAM-Med2D/calib_data/COCO_val2014_000000100811.jpg'
    image_array = cv2.imread(image_path)
    # print(img_array.shape)
    
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    image_array = (image_array - pixel_mean) / pixel_std
    
    h, w, _ = image_array.shape
    transforms = train_transforms(args.image_size, h, w)
    augments = transforms(image=image_array)
    image = augments['image'].float().unsqueeze(0).cuda()
    # print(image.shape)
    with torch.no_grad():
        image_embeddings = model.image_encoder(image)
        
def calib_distilled(args, q_model):
    calib_data = torch.load(args.distill_data_path)
    calib_data = calib_data.float().cuda()
    with torch.no_grad():
        image_embeddings = q_model.image_encoder(calib_data)
        
def calib_distilled2(args, q_model):
    calib_data = torch.load(args.distill_data_path)
    calib_data = calib_data.float().cuda()
    if calib_data.shape[1]==1:
        calib_data = calib_data.repeat(1,3,1,1)
        pixel_mean = torch.tensor([123.675/255, 116.28/255, 103.53/255]).view(1,3,1,1).cuda()
        pixel_std = torch.tensor([58.395/255, 57.12/255, 57.375/255]).view(1,3,1,1).cuda()
        calib_data = (calib_data - pixel_mean)/pixel_std
    with torch.no_grad():
        image_embeddings = q_model.image_encoder(calib_data)

def calib_load_data(args, q_model, load_calib_data_path):
    calib_data = torch.load(load_calib_data_path)
    print('loaded from', load_calib_data_path)
    calib_data = calib_data.float().cuda()
    if calib_data.shape[1]==1:
        calib_data = calib_data.repeat(1,3,1,1)
        pixel_mean = torch.tensor([123.675/255, 116.28/255, 103.53/255]).view(1,3,1,1).cuda()
        pixel_std = torch.tensor([58.395/255, 57.12/255, 57.375/255]).view(1,3,1,1).cuda()
        calib_data = (calib_data - pixel_mean)/pixel_std
        print('norm the data')
    with torch.no_grad():
        image_embeddings = q_model.image_encoder(calib_data)

def calib_load_data_gc(args, q_model, calib_pth_gray, calib_pth_color):
    data_gray = torch.load(calib_pth_gray)
    data_gray = data_gray.repeat(1,3,1,1)
    data_color = torch.load(calib_pth_color)
    
    data_calib = torch.cat((data_gray, data_color), dim=0).cuda()
    print(data_calib.shape)
    with torch.no_grad():
        image_embeddings = q_model.image_encoder(data_calib)

def quant_image_encoder(args, model, input_quant_params={}, weight_quant_params={}):
    # input
    input_quant_params_embed = deepcopy(input_quant_params)
    input_quant_params_embed['n_bits'] = 4

    # post-softmax
    input_quant_params_matmul2 = deepcopy(input_quant_params)
    if args.log2_quant:
        input_quant_params_matmul2['log_quant'] = True

    # SimQuant
    input_quant_params_channel = deepcopy(input_quant_params)
    input_quant_params_channel['channel_wise'] = args.PLNA_channel_wise

    module_dict={}
    for name, m in model.image_encoder.named_modules():
        module_dict[name] = m
        idx = name.rfind('.')
        if idx == -1:
            idx = 0
        father_name = name[:idx]
        if father_name in module_dict:
            father_module = module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        
        if isinstance(m, nn.Conv2d):
            # Embedding Layer
            idx = idx + 1 if idx != 0 else idx
            if 'embed' in name:
                new_m = QuantConv2d(
                    m.in_channels,
                    m.out_channels,
                    m.kernel_size,
                    m.stride,
                    m.padding,
                    m.dilation,
                    m.groups,
                    m.bias is not None,
                    input_quant_params_embed,
                    weight_quant_params
                )
            else:
                new_m = QuantConv2d(
                    m.in_channels,
                    m.out_channels,
                    m.kernel_size,
                    m.stride,
                    m.padding,
                    m.dilation,
                    m.groups,
                    m.bias is not None,
                    input_quant_params,
                    weight_quant_params
                )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, nn.Linear):
            # Linear Layer
            idx = idx + 1 if idx != 0 else idx
            if 'blocks' in name and ('Adapter' not in name):
                if 'qkv' in name or 'lin1' in name:
                    new_m = QuantLinear(m.in_features, m.out_features, input_quant_params_channel, weight_quant_params, True) 
                else:   
                    new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params, True)
            else:
                new_m = QuantLinear(m.in_features, m.out_features, input_quant_params, weight_quant_params, False)
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(father_module, name[idx:], new_m)
        elif isinstance(m, MatMul):
            # Matmul Layer
            idx = idx + 1 if idx != 0 else idx
            if 'matmul2' in name:
                new_m = QuantMatMul(input_quant_params_matmul2)
            else:
                new_m = QuantMatMul(input_quant_params)
            setattr(father_module, name[idx:], new_m)

    return model




def set_quant_state(model, input_quant=False, weight_quant=False):
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(input_quant, weight_quant)
            
def scale_reparameter(args, q_model):
    with torch.no_grad():
        module_dict={}
        q_model_slice = q_model.image_encoder.blocks
        for name, module in q_model_slice.named_modules():
            module_dict[name] = module
            idx = name.rfind('.')
            if idx == -1:
                idx = 0
            father_name = name[:idx]
            if father_name in module_dict:
                father_module = module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")

            # if 'norm1' in name or 'norm2' in name or 'norm' in name:
            if 'norm1' in name or 'norm2' in name:
                if 'norm1' in name:
                    next_module = father_module.attn.qkv
                elif 'norm2' in name:
                    next_module = father_module.mlp.lin1
                
                act_delta = next_module.input_quantizer.delta.reshape(-1)
                act_zero_point = next_module.input_quantizer.zero_point.reshape(-1)
                act_min = -act_zero_point * act_delta
                
                target_delta = torch.mean(act_delta)
                target_zero_point = torch.mean(act_zero_point)
                target_min = -target_zero_point * target_delta

                r = act_delta / target_delta
                b = act_min / r - target_min

                module.weight.data = module.weight.data / r
                module.bias.data = module.bias.data / r - b
               
                next_module.weight.data = next_module.weight.data * r
                if next_module.bias is not None:
                    next_module.bias.data = next_module.bias.data + torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)
                else:
                    next_module.bias = Parameter(torch.Tensor(next_module.out_features))
                    next_module.bias.data = torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)

                next_module.input_quantizer.channel_wise = False
                next_module.input_quantizer.delta = target_delta
                next_module.input_quantizer.zero_point = target_zero_point
                next_module.weight_quantizer.inited = False


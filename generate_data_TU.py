
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import RandomRotation
from skimage.measure import label, regionprops
from skimage.morphology import dilation, square

from quant_utils import *
from utils import *

from tensorboardX import SummaryWriter

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from visualization_TU import *
from torchvision.utils import draw_bounding_boxes


model_zoo = {'deit_tiny': 'deit_tiny_patch16_224',
            'deit_small': 'deit_small_patch16_224',
            'deit_base': 'deit_base_patch16_224',
            'swin_tiny': 'swin_tiny_patch4_window7_224',
            'swin_small': 'swin_small_patch4_window7_224',
            }


class AttentionMap:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.feature = output

    def remove(self):
        self.hook.remove()
        
def prompt_and_decoder_for_distill(args, boxes, model, image_embeddings):
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=boxes,
        masks=None,
    )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
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

def get_boxes_tensor_from_mask_tensor(mask_tensor):
    B, C, H, W = mask_tensor.shape
    new_box_tensor_list = []
    for i in range(B):
        mask_i = mask_tensor[i,:,:,:].squeeze(0)
        box = get_boxes_from_mask(mask_i.cpu()).cuda()
        new_box_tensor_list.append(box)
    box_tensor = torch.stack(new_box_tensor_list).cuda()
    return box_tensor   # B, C, 4


def TU_lab2SAM_lab_region(args, lab_batch_TU, TUlab_save_path='./png/show_target_lab_TU_', show_mask=True, min_region = 100):
    B, H, W = lab_batch_TU.shape
    if B != 1:
        print("error batch size!!!!")
        return
    lab_SAM_list = []
    lab_SAM_id_list = []
    
    c_list = torch.unique(lab_batch_TU.squeeze(0)).cpu().tolist()
   
    for ci in c_list[1:]:
        
        lab_bi = copy.deepcopy(lab_batch_TU.squeeze(0))
        lab_bi[lab_bi != ci] = int(0)
        lab_bi[lab_bi == ci] = int(1)
        
        labeled_mask = label(lab_bi.cpu(), connectivity=2)
        regions = regionprops(labeled_mask)
        
        for region in regions:
            lab_bi_rj = torch.zeros_like(lab_bi)
            if region.area > 80:
                lab_bi_rj[labeled_mask == region.label] = 1
                lab_bi_rj = lab_bi_rj.unsqueeze(0).unsqueeze(0)
                if H != args.image_size or W != args.image_size:
                    transform = transforms.Resize((args.image_size, args.image_size))
                    lab_bi_rj = transform(lab_bi_rj).to(torch.int64)
                lab_SAM_list.append(lab_bi_rj)
                lab_SAM_id_list.append(ci)
        
    box_list = []
    for lab_c_t_i in lab_SAM_list:
        B, C, H, W = lab_c_t_i.shape
        box = get_boxes_tensor_from_mask_tensor(lab_c_t_i.cpu()).cuda()
        box_list.append(box)
    
    if show_mask:
        # print('SAM gt list len:', len(lab_SAM_list))
        show_TU_label_BHW(args, lab_batch_TU, TUlab_save_path)
    
    return lab_SAM_list, box_list, lab_SAM_id_list
    


def show_sam_label_multi(args, gt_tensor_show, box_tensor_show):
    # 用于绘制sam的label，多个框的情况
    # gt_tensor_show.shape = [B,Class_num,H,W] box_tensor_show.shape = [B,Class_num,4]
    B, C, H, W = gt_tensor_show.shape
    if args.calib_batch_size!=1:
        n = int(args.calib_batch_size**0.5+0.999999)
        fig, ax = plt.subplots(n, n, figsize=(10, 10))
        for bi in range(args.calib_batch_size):
        # for j, (gt,box) in enumerate(zip([gts[i].squeeze(0) for i in ids], [boxes[i].squeeze(0) for i in ids])):
            # print(ax[j])
            gt = gt_tensor_show[bi,:,:,:]
            box = box_tensor_show[bi,:,:]
            gt_01 = torch.sum(gt, dim=0)
            gt_01[gt_01 != 0] = 1
            gt_01 = gt_01.unsqueeze(0)
            # print(gt_01.shape)  # torch.Size([1, 256, 256])
            a = bi//n
            b = bi%n
                       
            show_mask(gt_01.cpu().numpy(), ax[a][b])
            # show_mask(gt_tensor[bi,:,:,:].cpu().numpy(), ax[a][b])
            num_box = box.shape[0]
            for nj in range(num_box):
                box_j = box[nj,:]
                show_box(box_j, ax[a][b])  
        plt.savefig('show_box_gt.png')
    else:
        fig, ax = plt.subplots()
        gt_01 = torch.sum(gt_tensor_show.squeeze(0), dim=0)
        gt_01[gt_01 != 0] = 1
        gt_01 = gt_01.unsqueeze(0)
        show_mask(gt_01.cpu().numpy(), ax)
        box = box_tensor_show.squeeze(0)
        num_box = box.shape[0]
        for nj in range(num_box):
            box_j = box[nj,:]
            show_box(box_j, ax)
        plt.savefig('show_box_gt.png')
        
def show_TU_label_BHW(args, TU_lab, save_path='./png/show_target_lab_TU_'):
    B, H, W = TU_lab.shape
    for gi in range(B):
        label_np = TU_lab[gi,:,:].cpu().detach().numpy()
        fig, ax = plt.subplots(figsize=(5.12, 5.12))
        show_mask_TU(label_np, ax, 1)
        ax.set_xticks([])  # 隐藏x轴刻度
        ax.set_yticks([])  # 隐藏y轴刻度

        plt.savefig(save_path+str(gi)+'.png')
        plt.close()



def get_init_label(args, lab_num = 1):
    # 一个随机类别椭圆
    N, H, W =args.calib_batch_size, args.TU_img_size, args.TU_img_size
    init_label = torch.zeros(N, H, W).int()
    Rand_roate = RandomRotation(25)
    
    for b in range(N):
        for n in range(lab_num):
            label_class = random.randint(1,8)
            # label_class = 7
            c = [random.randint(100, 150), random.randint(100, 150)]
            aa = random.randint(10, 20)
            ur = np.random.uniform(1, 2.5)
            bb = aa * ur
            for i in range(H):
                for j in range(W):
                    if ((i - c[0])/aa)**2 + ((j - c[1])/bb)**2 < 1 :# and (i-Ctr[0])**2 + (j-Ctr[1])**2 < R ** 2:
                        init_label[b,i,j] = label_class
            tmp_lab = init_label[b,:,:].unsqueeze(0)
            init_label[b,:,:] = Rand_roate(tmp_lab).squeeze(0)
    torch.save(init_label,'./pth_save/init_label.pth')
    print("save init_label")
            
    return init_label.cuda()


def label_filter(args, TU_lab):
    # 过滤采样mask
    B,H,W = TU_lab.shape
    new_TU_lab_ls = []
    for bi in range(B):
        lab_bi = TU_lab[bi,:,:]
        new_mask_bi = torch.zeros_like(lab_bi)
        for cj in range(1,args.TU_num_classes):
            lab_bi_cj = lab_bi.cpu()
            lab_bi_cj[lab_bi==cj]=1
            lab_bi_cj[lab_bi!=cj]=0
            labeled_mask = label(lab_bi_cj, connectivity=2)
            regions = regionprops(labeled_mask)
            regions = sorted(regions, key=lambda x: x.area, reverse=True)
            
            count = 0
            for region in regions:
                if count>=2:
                    break
                new_mask_bi[labeled_mask == region.label] = cj
                count +=1
        new_TU_lab_ls.append(new_mask_bi)
    new_TU_lab = torch.stack(new_TU_lab_ls)
    return new_TU_lab

def label_filter_pro(args, TU_lab, TU_score):
    # 过滤采样mask，加入类别置信度
    B,H,W = TU_lab.shape
    new_TU_lab_ls = []
    for bi in range(B):
        lab_bi = TU_lab[bi,:,:]
        score_bi = TU_score[bi,:,:,:]
        new_mask_bi = torch.zeros_like(lab_bi)
        for cj in range(1,args.TU_num_classes):
            lab_bi_cj = lab_bi.cpu()
            lab_bi_cj[lab_bi==cj]=1
            lab_bi_cj[lab_bi!=cj]=0
            score_bi_cj = score_bi[cj,:,:]
            labeled_mask = label(lab_bi_cj, connectivity=2)
            regions = regionprops(labeled_mask, intensity_image=score_bi_cj.cpu().numpy())
            # 根据像素的平均分类置信度对区域进行排序
            label_avg_scores = [(region, region.mean_intensity) for region in regions]
            sorted_regions = sorted(label_avg_scores, key=lambda x: x[1], reverse=True)
            
            count = 0
            for region in sorted_regions:
                if count>=2:
                    break
                new_mask_bi[labeled_mask == region[0].label] = cj
                count +=1
        new_TU_lab_ls.append(new_mask_bi)
    new_TU_lab = torch.stack(new_TU_lab_ls)
    return new_TU_lab

def label_filter2(model_TU, img, args, TU_lab):
    # 过滤进化label
    B,H,W = TU_lab.shape
    new_TU_lab_ls = []
    
    with torch.no_grad():
        pred_mask = model_TU(img)
        pred_mask = torch.argmax(torch.softmax(pred_mask, dim=1), dim=1, keepdim=True)
        pred_mask = pred_mask.squeeze(1)
        
        for bi in range(B):
            lab_bi = copy.deepcopy(TU_lab[bi,:,:])
            new_mask_bi = torch.zeros_like(lab_bi)
            pred_mask_bi = pred_mask[bi,:,:]
            for cj in range(1,args.TU_num_classes):
                
                lab_bi_cj = torch.zeros_like(lab_bi).cpu()
                lab_bi_cj[lab_bi==cj]=1
                
                if cj == 8 and torch.sum(lab_bi_cj) > 3000:
                    # 对特别容易长得巨大的类别，严格限制面积
                    lab_bi_cj[pred_mask_bi!=cj] = 0
                    lab_bi_cj[pred_mask_bi==cj] = 1
                    # print('slim large region:'+str(cj))
                    
                if torch.sum(lab_bi_cj) > 6000:
                    lab_bi_cj[pred_mask_bi!=cj] = 0
                    lab_bi_cj[pred_mask_bi==cj] = 1
                    # print('slim large region:'+str(cj))
                
                labeled_mask = label(lab_bi_cj, connectivity=2)
                regions = regionprops(labeled_mask)
                regions = sorted(regions, key=lambda x: x.area, reverse=True)
                
                count = 0
                for region in regions:
                    # 限制每个类别最多取两个区域
                    if count>=2:
                        break
                    new_mask_bi[labeled_mask == region.label] = cj
                    count +=1
            new_TU_lab_ls.append(new_mask_bi)
        new_TU_lab = torch.stack(new_TU_lab_ls)
    return new_TU_lab

def label_filter_for_show(model_TU, img, args, TU_lab):
    B,H,W = TU_lab.shape
    new_TU_lab_ls = []
    
    with torch.no_grad():
        pred_mask = model_TU(img)
        pred_mask = torch.argmax(torch.softmax(pred_mask, dim=1), dim=1, keepdim=True)
        pred_mask = pred_mask.squeeze(1)
        for bi in range(B):
            lab_bi = copy.deepcopy(TU_lab[bi,:,:])
            new_mask_bi = torch.zeros_like(lab_bi)
            pred_mask_bi = pred_mask[bi,:,:]
            for cj in range(1,args.TU_num_classes):
                lab_bi_cj = torch.zeros_like(lab_bi).cpu()
                lab_bi_cj[lab_bi==cj]=1
                
                if torch.sum(lab_bi_cj) > 6000:
                    lab_bi_cj[pred_mask_bi!=cj] = 0
                    lab_bi_cj[pred_mask_bi==cj] = 1
                    print('slim large region:'+str(cj))
                
                labeled_mask = label(lab_bi_cj, connectivity=2)
                regions = regionprops(labeled_mask)
                regions = sorted(regions, key=lambda x: x.area, reverse=True)
                
                count = 0
                for region in regions:
                    if count>=2:
                        break
                    if region.area < 50:
                        break
                    new_mask_bi[labeled_mask == region.label] = cj
                    count +=1
            new_TU_lab_ls.append(new_mask_bi)
        new_TU_lab = torch.stack(new_TU_lab_ls)
    return new_TU_lab


def judge_area(TU_lab, class_id, area_threshold):
    class_lab = torch.zeros_like(TU_lab)
    class_lab[TU_lab==class_id]=1
    if torch.sum(class_lab) < area_threshold:
        return True
    else:
        return False
    


def generate_data2(args, p_model,writer):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True,warn_only=True)
    
    if not os.path.exists('./png_'+args.show_time):
         os.makedirs('./png_'+args.show_time)
    
    p_model.eval()
    
    # Hook the attention
    pixel_mean = torch.tensor([123.675/255, 116.28/255, 103.53/255]).view(1,3,1,1).cuda()
    pixel_std = torch.tensor([58.395/255, 57.12/255, 57.375/255]).view(1,3,1,1).cuda()
    hooks = []
    for m in p_model.image_encoder.blocks:
        hooks.append(AttentionMap(m.attn.matmul2))
        
    # 导入TransUNet
    config_vit = CONFIGS_ViT_seg[args.TU_vit_name]
    config_vit.n_classes = args.TU_num_classes
    config_vit.n_skip = args.TU_n_skip
    config_vit.patches.size = (args.TU_vit_patches_size, args.TU_vit_patches_size)
    if args.TU_vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.TU_img_size/args.TU_vit_patches_size), int(args.TU_img_size/args.TU_vit_patches_size))
    TU_net = ViT_seg(config_vit, img_size=args.TU_img_size, num_classes=config_vit.n_classes).cuda()
    snapshot = args.TU_ckpt_path
    TU_net.load_state_dict(torch.load(snapshot))
    TU_net.eval()
    print("*** TransUNet loaded ***")

    
    # Init Gaussian noise
    if args.img_channel == 3:
        img = torch.randn((args.calib_batch_size, 3, args.image_size, args.image_size)).cuda()
    else:
        img = torch.randn((args.calib_batch_size, 1, args.image_size, args.image_size)).cuda()

    save_img(args, img, './png_'+args.show_time+'/show_distil_img_init.png')
    img.requires_grad = True
    
    gt_list = []
    TU_lab = get_init_label(args)
    gt_list, gt_box_list, gt_id_list= TU_lab2SAM_lab_region(args, TU_lab, './png_'+args.show_time+'/show_init_lab_TU_', True, 50)
        

    # Init optimizer
    optimizer = optim.Adam([img], lr=args.ZQ_lr, betas=[0.5, 0.9], eps=1e-8)

    # param For TV loss
    if args.img_channel==1:
        var_pred = random.uniform(50, 55)  # for batch_size 32 (2500, 3000)
    else:
        var_pred = random.uniform(150, 155)  # for batch_size 32 (2500, 3000)
        
    criterion = FocalDiceloss_IoULoss()
    criterion_TU = nn.CrossEntropyLoss()
    
    # Train for two epochs
    for lr_it in range(3):
        if lr_it == 0:
            iterations_per_layer = 500
            lim = 15
        elif lr_it == 1:
            iterations_per_layer = 500
            lim = 15
        else:
            iterations_per_layer = 500
            lim = 30

        lr_scheduler = lr_cosine_policy(args.ZQ_lr, 100, iterations_per_layer)
        
        gt_ud_cnt = 0
        new_TU_lab_list = [TU_lab]

        with tqdm(range(iterations_per_layer)) as pbar:
            for itr in pbar:
                pbar.set_description(f"Epochs {lr_it+1}/{2}")

                # Learning rate scheduling
                lr_scheduler(optimizer, itr, itr)
                
                if lr_it == 0 and itr <= 800:
                    with torch.no_grad():
                        transform = transforms.Resize((args.TU_img_size, args.TU_img_size))
                        img_224 = transform(img)
                        pred_TU = TU_net(img_224)
                        TU_output_score = torch.softmax(pred_TU, dim=1)
                        
                        new_TU_lab = copy.deepcopy(TU_lab)
                        TU_pred = torch.argmax(torch.softmax(pred_TU, dim=1), dim=1, keepdim=True)  # torch.Size([1, 1, 224, 224])
                        
                        TU_pred = TU_pred.squeeze(1)
                        TU_pred = label_filter_pro(args, TU_pred, TU_output_score)
                        
                        
                        for class_id in range(1,args.TU_num_classes):               
                            if judge_area(new_TU_lab, class_id, 6000):
                                new_TU_lab[TU_pred == class_id] = class_id
                        TU_lab = copy.deepcopy(new_TU_lab)

                        TU_lab = label_filter2(TU_net, img_224, args, TU_lab)
                        
                        fake_label_vis = label_filter_for_show(TU_net, img_224, args, TU_lab)
                        
                        gt_list, box_list, gt_id_list = TU_lab2SAM_lab_region(args, TU_lab, './png_'+args.show_time+'/show_filter_lab_TU_ep_'+str(lr_it)+'_'+str(itr)+'_', False)
                        if itr % 10==0:
                            # 可视化
                            show_TU_label_BHW(args, fake_label_vis, './png_'+args.show_time+'/PPE_vis_'+str(lr_it)+'_'+str(itr)+'_')
                            save_img(args, img, './png_'+args.show_time+'/show_distil_img_'+str(lr_it)+'_'+str(itr)+'.png')
                            show_box_mask(args, img, box_list, gt_list, gt_id_list, save_path = './png_'+args.show_time+'/', id = str(lr_it)+'_'+str(itr))
                        gt_ud_cnt = 0
          
                if gt_ud_cnt < len(gt_list):
                    gt_ud_cnt += 1
                else:
                    gt_ud_cnt = 1
                
                gt_tensor = torch.cat(gt_list, dim=0)


                # Apply random jitter offsets (from DeepInversion[1])
                # [1] Yin, Hongxu, et al. "Dreaming to distill: Data-free knowledge transfer via deepinversion.", CVPR2020.
                off = random.randint(-lim, lim)
                img_jit = torch.roll(img, shifts=(off, off), dims=(2, 3))
                gt_tensor_jit = torch.roll(gt_tensor, shifts=(off, off), dims=(2, 3))
                TU_lab_jit = torch.roll(TU_lab, shifts=(off, off), dims=(1, 2))
                # Flipping
                flip = random.random() > 0.5
                if flip:
                    img_jit = torch.flip(img_jit, dims=(3,))
                    gt_tensor_jit = torch.flip(gt_tensor_jit, dims=(3,))
                    TU_lab_jit = torch.flip(TU_lab_jit, dims=(2,))
                box_tensor_jit = get_boxes_tensor_from_mask_tensor(gt_tensor_jit)
                # box_tensor_jit = box_tensor
                               
                
                # Forward pass
                optimizer.zero_grad()
                p_model.zero_grad()

                if args.img_channel==1:
                    img_jit_3C = img_jit.repeat(1,3,1,1)
                else:
                    img_jit_3C = img_jit
                img_jit_3C = (img_jit_3C - pixel_mean)/pixel_std
                
                image_embeddings = p_model.image_encoder(img_jit_3C)
                # print(image_embeddings.shape)   # torch.Size([1, 256, 16, 16])
                image_embeddings = image_embeddings.repeat(box_tensor_jit.shape[0],1,1,1)
                masks, low_res_masks, iou_predictions = prompt_and_decoder_for_distill(args, box_tensor_jit, p_model, image_embeddings)
                # print(masks.shape)      # torch.Size([1, 1, 256, 256])
                loss_seg = criterion(masks, gt_tensor_jit, iou_predictions)
                
                loss_tv = torch.norm(get_image_prior_losses(img_jit) - var_pred)

                loss_entropy = 0
                
                for itr_hook in range(len(hooks)):
                    # Hook attention
                    attention = hooks[itr_hook].feature
                    BnH, N, E_nH = attention.shape
                    nH = 12
                    attention = attention.reshape(BnH//nH, nH, N, E_nH)
                    attention_p = attention.mean(dim=1)
                    sims = torch.cosine_similarity(attention_p.unsqueeze(1), attention_p.unsqueeze(2), dim=3)

                    # Compute differential entropy
                    kde = KernelDensityEstimator(sims.view(BnH//nH, -1))
                    start_p = sims.min().item()
                    end_p = sims.max().item()
                    x_plot = torch.linspace(start_p, end_p, steps=10).repeat(BnH//nH, 1).cuda()
                    kde_estimate = kde(x_plot)
                    dif_entropy_estimated = differential_entropy(kde_estimate, x_plot)
                    loss_entropy -= dif_entropy_estimated
                    
                # TU class loss
                transform = transforms.Resize((args.TU_img_size, args.TU_img_size))
                img_jit_224 = transform(img_jit)
                target_labels_224 = transform(TU_lab_jit.unsqueeze(0)).to(torch.int64)
                target_labels_224 = target_labels_224.squeeze(0)
                
                output_TU = TU_net(img_jit_224)
                loss_class = criterion_TU(output_TU, target_labels_224[:].long())
                
               
                total_loss = args.para_class*loss_class + args.para_seg*loss_seg + args.para_PSE * loss_entropy + args.para_tv * loss_tv    # (0.5,1,0.05,0.005)
                
                # Do image update
                total_loss.backward()
                optimizer.step()

                # Clip color outliers
                img.data = clip3(img.data)
                
                pred_masks = torch.sigmoid(masks)
                pred_masks[pred_masks > 0.5] = int(1)
                pred_masks[pred_masks <= 0.5] = int(0)
                
                # Summary Writer
                if itr % 5 ==0:
                    img_show=(img - img.min()) / (img.max() - img.min()+1e-9)
                    writer.add_images(args.SW_log_dir+'distill'+'/data', img_show,itr)
                    TU_output_show = torch.argmax(torch.softmax(output_TU, dim=1), dim=1, keepdim=True)
                    B,C,H,W=pred_masks.shape
                    show_pred_box_list = []
                    for i in range(B):
                        pred = pred_masks[i,:,:,:].repeat(3,1,1).to(torch.uint8)*255  # (3,256,256)
                        pred_with_boxes = draw_bounding_boxes(pred, box_tensor_jit[i,:,:], colors="red", width=2)
                        show_pred_box_list.append(pred_with_boxes)
                    show_pred_box = torch.stack(show_pred_box_list)
                    writer.add_images(args.SW_log_dir+'distill'+'/gt_box', show_pred_box, itr)
                    
                    writer.add_images(args.SW_log_dir+'distill'+'/TU_Prediction', TU_output_show * 25, itr)
                if (itr+1) %50 ==0:
                    save_img(args, img, './png_'+args.show_time+'/show_distil_img_'+str(lr_it)+'_'+str(itr+1)+'.png')
                    
                    
    torch.save(img, args.distill_data_path)
    if not os.path.exists('visualize/distill_result'):
        os.makedirs('visualize/distill_result')
    torch.save(box_list, 'visualize/distill_result/box_list_'+args.show_time+'.pth')
    
    print("***visualization***")
    try:
        img = img.detach()
        p_model.eval()
        for bi in range(img.shape[0]):
            img_i = copy.deepcopy(img[bi,:,:,:]).unsqueeze(0)
            img_i = (img_i - img_i.min()) / (img_i.max() - img_i.min()+1e-9)
            
            image_HWC = img_i.squeeze(0).permute(1,2,0).cpu().numpy()
            # print(img_i.shape)
            pixel_mean = torch.tensor([123.675/255, 116.28/255, 103.53/255]).view(1,3,1,1).cuda()
            pixel_std = torch.tensor([58.395/255, 57.12/255, 57.375/255]).view(1,3,1,1).cuda()
            if args.img_channel == 1:
                img_i_norm = (img_i.repeat(1,3,1,1) - pixel_mean)/pixel_std
                image_RGB = cv2.cvtColor(image_HWC, cv2.COLOR_BGR2RGB)
            else:
                img_i_norm = (img_i - pixel_mean)/pixel_std
                image_RGB = image_HWC
            with torch.no_grad():
                image_embeddings = p_model.image_encoder(img_i_norm)
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(image_RGB)
            ax[1].imshow(image_RGB)
            
            for bj, (box, gt_id) in enumerate(zip(box_list, gt_id_list)):
                masks, low_res_masks, iou_predictions = prompt_and_decoder_for_distill(args, box, p_model, image_embeddings)
                pred_masks = torch.sigmoid(masks)
                pred_masks[pred_masks > 0.5] = int(1)
                pred_masks[pred_masks <= 0.5] = int(0)
                pred_masks = pred_masks.cpu().detach()
                show_boxes(box, None, ax[0], args.image_size, args.image_size, args.image_size)
                show_boxes(box, None, ax[1], args.image_size, args.image_size, args.image_size)
                show_mask(pred_masks, ax[1], True, 0.9, gt_id-1)
            save_path = 'visualize/distill_result/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            ax[0].axis('off')
            ax[1].axis('off')
            plt.savefig(save_path+'distil_data_'+str(bi)+'_'+args.show_time+'.png')
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return img.detach()



def differential_entropy(pdf, x_pdf):  
    # pdf is a vector because we want to perform a numerical integration
    pdf = pdf + 1e-4
    f = -1 * pdf * torch.log(pdf)
    # Integrate using the composite trapezoidal rule
    ans = torch.trapz(f, x_pdf, dim=-1).mean()  
    return ans


def get_image_prior_losses(inputs_jit):
    # Compute total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    return loss_var_l2



def clip3(image_tensor, use_fp16=False):
    B,C,H,W = image_tensor.shape
    m = sum([0.485, 0.456, 0.406])/3
    s = sum([0.229, 0.224, 0.225])/3
    image_tensor = torch.clamp(image_tensor, -m / s, (1 - m) / s)
        
    return image_tensor


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)

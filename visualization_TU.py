from torch.utils.data import DataLoader
from tqdm import tqdm
from quant import *
from utils import *
from utils_TU import test_single_volume, inference_single_volume
from matplotlib import pyplot as plt
import copy


def set_ax_frame_line(ax,nums):
    bwith = 0.2
    for i in range(nums):
        ax[i].spines['bottom'].set_linewidth(bwith)#图框下边
        ax[i].spines['left'].set_linewidth(bwith)#图框左边
        ax[i].spines['top'].set_linewidth(bwith)#图框上边
        ax[i].spines['right'].set_linewidth(bwith)#图框右边
        ax[i].axis('off')
        
    
def show_mask_TU(mask, ax, alpha = 0.5):
    mask = mask.astype(np.int16)
    category_ids = np.unique(mask)
    
    category_colors = [np.array([252 / 255, 0 / 255, 0 / 255, alpha]),      # 红，主动脉
                       np.array([251 / 255, 252 / 255, 0 / 255, alpha]),    # 黄，胆囊
                       np.array([0 / 255, 255 / 255, 255 / 255, alpha]),    # 青， 肾脏L
                       np.array([160 / 255, 32 / 255, 240 / 255, alpha]),   # 紫， 肾脏R
                       np.array([255 / 255, 0 / 255, 255 / 255, alpha]),    # 粉， 肝
                       np.array([255 / 255, 128 / 255, 0 / 255, alpha]),    # 橙， 胰腺
                       np.array([124 / 255, 252 / 255, 0 / 255, alpha]),    # 绿， 脾
                       np.array([3 / 255, 84 / 255, 249 / 255, alpha])]     # 蓝， 胃
    
    # print(category_ids)
   
    h, w = mask.shape[-2:]
    
    for ci in category_ids[1:]:
        mask_i = mask.copy()
        mask_i = mask_i.reshape(h, w, 1)
        mask_i[mask != ci] = 0
        mask_i[mask == ci] = 1
        color = category_colors[ci-1]
        mask_image = mask_i * color.reshape(1, 1, -1)
        ax.imshow(mask_image, cmap='gray')

def show_img_mask(args, model, batch_nums,save_name=''):
    save_path='./vis_png/'+args.dataset+'/'+save_name+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.eval()
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        if i_batch > batch_nums:
            break
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        
        prediction = inference_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=None, case=case_name, z_spacing=args.z_spacing)
        print(prediction.shape) # (148, 512, 512)
        image_np, label_np = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
        # print(image_np.shape, label_np.shape)   #(148, 512, 512) (148, 512, 512)
        z_index, _, _ = np.where(label_np>0)
        z_index = np.unique(z_index)
        
        for i, id in enumerate(z_index):
            if i%2 ==0:
                image_si = image_np[id,:,:]
                image_si = (image_si-image_si.min())/np.clip(image_si.max()-image_si.min(), a_min=1e-8, a_max=None)
                image_si = np.repeat(np.expand_dims(image_si, axis=2), repeats=3, axis=2)
                label_si = label_np[id,:,:]
                pred_si = prediction[id,:,:]
                
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(image_si)
                ax[1].imshow(image_si)
                ax[2].imshow(image_si)
                show_mask(label_si, ax[1])
                show_mask(pred_si, ax[2])
                set_ax_frame_line(ax,3)
                
                plt.savefig(save_path+'bs'+str(i_batch)+'_s'+str(id)+'.png')
                plt.close()



def show_random_mask():
    random_mask = np.random.randint(0,8,size=[224,224])
    # print(random_mask)
    fig, ax = plt.subplots()
    show_mask(random_mask, ax, 1)
    ax.axis('off')
    plt.savefig('png/random_mask2.png')
    
    
def show_pth_label():
    pth_label = torch.load('/home/zhangjing/TransUNet_Proj/TransUNet_MSD/pth/target_labels.pth')
    print(pth_label.shape)
    label_np = pth_label.squeeze(0).cpu().detach().numpy()
    fig, ax = plt.subplots(figsize=(5.12, 5.12))
    show_mask(label_np, ax, 1)
    ax.axis('off')
    plt.savefig('./png/MSD_prediction3.png')
    
def show_pth_img_label():
    pth_img = torch.load('/home/zhangjing/TransUNet_Proj/TransUNet_MSD/generated_data/distill_data_Synapse_bs1_sm_lab.pth')
    pth_label = torch.load('/home/zhangjing/TransUNet_Proj/TransUNet_MSD/target_labels.pth')
    print(pth_label.shape)
    image_HWC = pth_img.squeeze(0).permute(1,2,0).cpu().numpy()
    _,_,C = image_HWC.shape
    if C==1:
        image_HWC = np.repeat(image_HWC, repeats=3, axis=2)
    # np.set_printoptions(threshold=np.inf)
    # print(image_HWC)
    # plt.figure(figsize=(5,5))
    image_HWC = (image_HWC - image_HWC.min()) / np.clip(image_HWC.max() - image_HWC.min(), a_min=1e-8, a_max=None)
    label_np = pth_label.squeeze(0).cpu().detach().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(image_HWC)
    ax[0].axis('off')
    ax[1].imshow(image_HWC)
    show_mask(label_np, ax[1])
    ax[1].axis('off')
    plt.savefig('generated_show.png')
    
def show_img_mask2(args, model1, model2, batch_nums,save_name=''):
    save_path='./vis_png/'+args.dataset+'/'+save_name+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model1.eval()
    model2.eval()
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        if i_batch > batch_nums:
            break
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        # print(image.shape)    # torch.Size([1, 148, 512, 512])
        prediction1 = inference_single_volume(image, label, model1, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=None, case=case_name, z_spacing=args.z_spacing)
        prediction1 = copy.deepcopy(prediction1)
        prediction2 = inference_single_volume(image, label, model2, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=None, case=case_name, z_spacing=args.z_spacing)
        prediction2 = copy.deepcopy(prediction2)
        
        print(prediction1.shape) # (148, 512, 512)
        image_np, label_np = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
        z_index, _, _ = np.where(label_np>0)
        z_index = np.unique(z_index)
        
        for i, id in enumerate(z_index):
            if i%1 ==0:
                image_si = image_np[id,:,:]
                image_si = (image_si-image_si.min())/np.clip(image_si.max()-image_si.min(), a_min=1e-8, a_max=None)
                image_si = np.repeat(np.expand_dims(image_si, axis=2), repeats=3, axis=2)
                label_si = label_np[id,:,:]
                pred_si1 = prediction1[id,:,:]
                pred_si2 = prediction2[id,:,:]
                
                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                ax[0].imshow(image_si)
                ax[1].imshow(image_si)
                ax[2].imshow(image_si)
                ax[3].imshow(image_si)
                show_mask(label_si, ax[1])
                show_mask(pred_si1, ax[2])
                show_mask(pred_si2, ax[3])
                set_ax_frame_line(ax,4)
                plt.savefig(save_path+'bs'+str(i_batch)+'_s'+str(id)+'.png')
                plt.close()
                
def show_noise():
    import torch
    from matplotlib import pyplot as plt
    import numpy as np
    img = torch.randn((224, 224))
    img = (img - img.min()) / (img.max() - img.min())
    fig, ax = plt.subplots()
    img = img.cpu().detach().numpy()
    img = np.repeat(np.expand_dims(img, axis=2), repeats=3, axis=2)

    ax.imshow(img)
    ax.axis('off')
    plt.savefig('noize224')
    
    import torch
    from matplotlib import pyplot as plt
    import numpy as np
    img = torch.randn((224, 224, 3))
    img = (img - img.min()) / (img.max() - img.min())
    fig, ax = plt.subplots()
    img = img.cpu().detach().numpy()

    ax.imshow(img)
    ax.axis('off')
    plt.savefig('png/noize224_3')
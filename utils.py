from albumentations.pytorch import ToTensorV2
import cv2
import albumentations as A
import torch
import numpy as np
from torch.nn import functional as F
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import random
import torch.nn as nn
import logging
import os
from matplotlib.patches import Polygon
import copy

def SW_show_img(args, writer, img_ori):
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1,3,1,1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1,3,1,1)
    img = 1
    img_show=(img - img.min()) / (img.max() - img.min()+1e-9)
    writer.add_images
    


def get_boxes_from_mask(mask, box_num=1, std = 0.1, max_pixel = 5):
    """
    Args:
        mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
        box_num: Number of bounding boxes, default is 1.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor.
    """
    # print(mask.shape)
    # exit()
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
    label_img = label(mask)
    regions = regionprops(label_img)

    # Iterate through all regions and get the bounding box coordinates
    boxes = [tuple(region.bbox) for region in regions]

    # If the generated number of boxes is greater than the number of categories,
    # sort them by region area and select the top n regions
    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]

    # If the generated number of boxes is less than the number of categories,
    # duplicate the existing boxes
    elif len(boxes) < box_num:
        num_duplicates = box_num - len(boxes)
        boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]

    # Perturb each bounding box with noise
    noise_boxes = []
    for box in boxes:
        y0, x0,  y1, x1  = box
        width, height = abs(x1 - x0), abs(y1 - y0)
        # Calculate the standard deviation and maximum noise value
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * 5))
         # Add random noise to each coordinate
        try:
            noise_x = np.random.randint(-max_noise, max_noise)
        except:
            noise_x = 0
        try:
            noise_y = np.random.randint(-max_noise, max_noise)
        except:
            noise_y = 0
        x0, y0 = x0 + noise_x, y0 + noise_y
        x1, y1 = x1 + noise_x, y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))
    return torch.as_tensor(noise_boxes, dtype=torch.float)


def select_random_points(pr, gt, point_num = 9):
    """
    Selects random points from the predicted and ground truth masks and assigns labels to them.
    Args:
        pred (torch.Tensor): Predicted mask tensor.
        gt (torch.Tensor): Ground truth mask tensor.
        point_num (int): Number of random points to select. Default is 9.
    Returns:
        batch_points (np.array): Array of selected points coordinates (x, y) for each batch.
        batch_labels (np.array): Array of corresponding labels (0 for background, 1 for foreground) for each batch.
    """
    pred, gt = pr.data.cpu().numpy(), gt.data.cpu().numpy()
    error = np.zeros_like(pred)
    error[pred != gt] = 1

    # error = np.logical_xor(pred, gt)
    batch_points = []
    batch_labels = []
    for j in range(error.shape[0]):
        one_pred = pred[j].squeeze(0)
        one_gt = gt[j].squeeze(0)
        one_erroer = error[j].squeeze(0)

        indices = np.argwhere(one_erroer == 1)
        if indices.shape[0] > 0:
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        else:
            indices = np.random.randint(0, 256, size=(point_num, 2))
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        selected_indices = selected_indices.reshape(-1, 2)

        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]
            if one_pred[x,y] == 0 and one_gt[x,y] == 1:
                label = 1
            elif one_pred[x,y] == 1 and one_gt[x,y] == 0:
                label = 0
            else:
                label = -1
            points.append((y, x))   #Negate the coordinates
            labels.append(label)

        batch_points.append(points)
        batch_labels.append(labels)
    return np.array(batch_points), np.array(batch_labels)


def init_point_sampling(mask, get_point=1):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
     # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:,::-1]
    bg_coords = np.argwhere(mask == 0)[:,::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices], dtype=torch.int)
        return coords, labels
    

def train_transforms(img_size, ori_h, ori_w):
    transforms = []
    if ori_h < img_size and ori_w < img_size:
        transforms.append(A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1.)

def train_transforms2(img_size):
    transforms = []
    # if ori_h < img_size and ori_w < img_size:
    #     # transforms.append(A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    #     transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_LINEAR))
    # else:
    transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1.)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def generate_point(masks, labels, low_res_masks, batched_input, point_num):
    masks_clone = masks.clone()
    masks_sigmoid = torch.sigmoid(masks_clone)
    masks_binary = (masks_sigmoid > 0.5).float()

    low_res_masks_clone = low_res_masks.clone()
    low_res_masks_logist = torch.sigmoid(low_res_masks_clone)

    points, point_labels = select_random_points(masks_binary, labels, point_num = point_num)
    batched_input["mask_inputs"] = low_res_masks_logist
    batched_input["point_coords"] = torch.as_tensor(points)
    batched_input["point_labels"] = torch.as_tensor(point_labels)
    batched_input["boxes"] = None
    return batched_input


def setting_prompt_none(batched_input):
    batched_input["point_coords"] = None
    batched_input["point_labels"] = None
    batched_input["boxes"] = None
    return batched_input


def draw_boxes(img, boxes):
    img_copy = np.copy(img)
    for box in boxes:
        cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return img_copy

def show_mask(mask, ax, order_color=False, alpha=0.7, color_id=0):
    # color_list = []
    color_list = [np.array([252 / 255, 0 / 255, 0 / 255, alpha]),      # 红，主动脉
                  np.array([251 / 255, 252 / 255, 0 / 255, alpha]),    # 黄，胆囊
                  np.array([0 / 255, 255 / 255, 255 / 255, alpha]),    # 青， 肾脏L
                  np.array([160 / 255, 32 / 255, 240 / 255, alpha]),   # 紫， 肾脏R
                  np.array([255 / 255, 0 / 255, 255 / 255, alpha]),    # 粉， 肝
                  np.array([255 / 255, 128 / 255, 0 / 255, alpha]),    # 橙， 胰腺
                  np.array([124 / 255, 252 / 255, 0 / 255, alpha]),    # 绿， 脾
                  np.array([3 / 255, 84 / 255, 249 / 255, alpha]),     # 蓝， 胃
                  np.array([93 / 255, 164 / 255, 229 / 255, alpha]),   # 浅蓝
                  np.array([229 / 255, 93 / 255, 93 / 255, alpha]),    # 玫红
                  np.array([145 / 255, 116 / 255, 226 / 255, alpha]),    # 雪青
                  np.array([222 / 255, 120 / 255, 203 / 255, alpha])]     # 淡粉
    if order_color:
        # color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        # color = random.choice(color_list)
        color = color_list[color_id%len(color_list)]
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, cmap='gray')
    
def show_boxes(boxes, pad, ax, ori_w, ori_h, image_size):
    boxes = boxes.squeeze().cpu().numpy()
    x0, y0, x1, y1 = boxes
    if pad is not None:
        x0_ori = int((x0 - pad[1]) + 0.5)
        y0_ori = int((y0 - pad[0]) + 0.5)
        x1_ori = int((x1 - pad[1]) + 0.5)
        y1_ori = int((y1 - pad[0]) + 0.5)
    else:
        x0_ori = int(x0 * ori_w / image_size) 
        y0_ori = int(y0 * ori_h / image_size) 
        x1_ori = int(x1 * ori_w / image_size) 
        y1_ori = int(y1 * ori_h / image_size)
    boxes = [(x0_ori, y0_ori, x1_ori, y1_ori)]
    
    # 绘制box
    for box in boxes:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="red", facecolor=(0, 0, 0, 0), lw=1.5))
        
def show_box(box, ax):
    # print(box.shape)
    # exit()
    box = box.cpu().numpy()
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="red", facecolor=(0, 0, 0, 0), lw=1))
    
def show_points(points, pad, ori_h, ori_w, image_size, ax):
    point_coords, point_labels = points[0].squeeze(0).cpu().numpy(),  points[1].squeeze(0).cpu().numpy()
    point_coords = point_coords.tolist()
    if pad is not None:
        ori_points = [[int((x * ori_w / image_size)) , int((y * ori_h / image_size))]if l==0 else [x - pad[1], y - pad[0]]  for (x, y), l in zip(point_coords, point_labels)]
    else:
        ori_points = [[int((x * ori_w / image_size)) , int((y * ori_h / image_size))] for x, y in point_coords]

    for point, label in zip(ori_points, point_labels):
        x, y = map(int, point)
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        dot_format = ['*', 'r'] if label==1 else ['o', 'b']
        ax.scatter(x,y,marker=dot_format[0],c=dot_format[1])
        # plt.plot()
        
def show_conters(masks, ax):
    # print(type(masks))
    
    _,_,H,W = masks.shape
    masks = masks.view(H,W).cpu().numpy()
    np.set_printoptions(threshold=np.inf)
    # print(masks)
    contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历所有轮廓并在图像上绘制
    for contour in contours:
    # 将轮廓点转换为整数坐标
        # print(contour.shape)
        # print(contour)
        contour = contour.astype(np.int32).squeeze(1)
        # x = contour[:,0]
        # y = contour[:,1]
        # print(x.shape)
        # exit()
    # 创建一个Polygon对象并添加到图像上
        if contour.shape[0]>5:
            # print(contour.shape)
            polygon = Polygon(contour[::3,:], closed=True, fill=False, edgecolor=(0,1,0,0.5), linewidth=1)
            ax.add_patch(polygon)
        # cv2.drawContours(ax, [contour], 0, (0, 255, 0), 3)

def save_masks2(args, image, labels, preds, save_path, img_name, image_size, original_size, pad=None,  boxes=None, points=None, visual_prompt=True, folder_name='dataset_name'):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # image = image.cpu().numpy()
    
    image_HWC = image.squeeze(0).permute(1,2,0).cpu().numpy()
    labels = labels.cpu()
    preds = preds.cpu()
    # print(image.shape)
    # print(original_size)
    ori_h, ori_w = original_size
    h, w = ori_h[0].item(), ori_w[0].item()
    # print(h, w)
    # print(image_HWC.shape)
    
    transform_img = A.Resize(ori_h, ori_w, interpolation=cv2.INTER_LINEAR)
    # image_HWC = cv2.resize(image_HWC, (w, h),interpolation=cv2.INTER_LINEAR)
    image_HWC = transform_img(image=image_HWC)['image']
    # print(image_HWC.shape)
    image_RGB = cv2.cvtColor(image_HWC, cv2.COLOR_BGR2RGB)
    
    ax[0].imshow(image_RGB)
    
    if boxes is not None:
        show_boxes(boxes, pad, ax[0], ori_w, ori_h, image_size)
    if points is not None:
        show_points(points, pad, ori_h, ori_w, image_size, ax[0])
    # 绘制gt
    # show_mask(labels, ax[0])
    # show_conters(labels, ax[0])
    ##########
    preds = torch.sigmoid(preds)
    preds[preds > 0.5] = int(1)
    preds[preds <= 0.5] = int(0)

    ax[1].imshow(image_RGB)
    if boxes is not None:
        show_boxes(boxes, pad, ax[1], ori_w, ori_h, image_size)
    if points is not None:
        show_points(points, pad, ori_h, ori_w, image_size, ax[1])
    show_mask(preds, ax[1])
    show_conters(labels, ax[1])
    
    
    # save_path = os.path.join(args.save_pred_path, folder_name)
    save_path = os.path.join(args.save_pred_path, args.test_mode, args.show_time, folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # print(save_path)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.savefig(os.path.join(save_path, img_name))
    plt.close()
    
def save_img(args, img_gray, save_dir=''):
    img_save = img_gray.data
    if img_save.shape[1] == 1:
        img_save = img_save.repeat(1,3,1,1)
    image_HWC = img_save.squeeze(0).permute(1,2,0).cpu().numpy()
    image_HWC = (image_HWC - image_HWC.min()) / np.clip(image_HWC.max() - image_HWC.min(), a_min=1e-8, a_max=None)
    plt.imsave(save_dir, image_HWC)

def show_box_mask(args, img, box_list, gt_list, gt_id_list, save_path = './png/', id='0'):
    with torch.no_grad():
        img = img.detach()
        for bi in range(img.shape[0]):
            img_i = copy.deepcopy(img[bi,:,:,:]).unsqueeze(0)
           
            img_i = (img_i - img_i.min()) / (img_i.max() - img_i.min()+1e-9)
            fig, ax = plt.subplots(1, 1)
            # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            image_HWC = img_i.squeeze(0).permute(1,2,0).cpu().numpy()
            image_RGB = cv2.cvtColor(image_HWC, cv2.COLOR_BGR2RGB)
            # ax[0].imshow(image_RGB)
            # ax[1].imshow(image_RGB)
            ax.imshow(image_RGB)
            
            for bj, (box, gt) in enumerate(zip(box_list, gt_list)):
                gt = gt.cpu().detach()
                # show_boxes(box, None, ax[0], args.image_size, args.image_size, args.image_size)
                # show_boxes(box, None, ax[1], args.image_size, args.image_size, args.image_size)
                show_boxes(box, None, ax, args.image_size, args.image_size, args.image_size)
                show_mask(gt, ax, True, 0.8, gt_id_list[bj]-1)
                # show_mask(gt, ax[1], True, 0.9, bj)
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # ax[0].axis('off')
            # ax[1].axis('off')
            ax.axis('off')
            
            plt.savefig(save_path+'box_mask_dev_'+id+'_'+str(bi)+'.png')
            plt.close()

def save_masks(preds, save_path, mask_name, image_size, original_size, pad=None,  boxes=None, points=None, visual_prompt=False):
    ori_h, ori_w = original_size

    preds = torch.sigmoid(preds)
    preds[preds > 0.5] = int(1)
    preds[preds <= 0.5] = int(0)

    mask = preds.squeeze().cpu().numpy()
    mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    if visual_prompt: #visualize the prompt
        if boxes is not None:
            boxes = boxes.squeeze().cpu().numpy()

            x0, y0, x1, y1 = boxes
            if pad is not None:
                x0_ori = int((x0 - pad[1]) + 0.5)
                y0_ori = int((y0 - pad[0]) + 0.5)
                x1_ori = int((x1 - pad[1]) + 0.5)
                y1_ori = int((y1 - pad[0]) + 0.5)
            else:
                x0_ori = int(x0 * ori_w / image_size) 
                y0_ori = int(y0 * ori_h / image_size) 
                x1_ori = int(x1 * ori_w / image_size) 
                y1_ori = int(y1 * ori_h / image_size)

            boxes = [(x0_ori, y0_ori, x1_ori, y1_ori)]
            mask = draw_boxes(mask, boxes)

        if points is not None:
            point_coords, point_labels = points[0].squeeze(0).cpu().numpy(),  points[1].squeeze(0).cpu().numpy()
            point_coords = point_coords.tolist()
            if pad is not None:
                ori_points = [[int((x * ori_w / image_size)) , int((y * ori_h / image_size))]if l==0 else [x - pad[1], y - pad[0]]  for (x, y), l in zip(point_coords, point_labels)]
            else:
                ori_points = [[int((x * ori_w / image_size)) , int((y * ori_h / image_size))] for x, y in point_coords]

            for point, label in zip(ori_points, point_labels):
                x, y = map(int, point)
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                mask[y, x] = color
                cv2.drawMarker(mask, (x, y), color, markerType=cv2.MARKER_CROSS , markerSize=7, thickness=2)  
    os.makedirs(save_path, exist_ok=True)
    mask_path = os.path.join(save_path, f"{mask_name}")
    cv2.imwrite(mask_path, np.uint8(mask))
    
#Loss funcation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss


class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):
    
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss

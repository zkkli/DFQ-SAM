from tensorboardX import SummaryWriter
import torch
import copy

writer = SummaryWriter('./about_label_log')

distill_data = torch.load('./generated_data/distill_data_store2_dice0_756_hd31_1.pth')
print(type(distill_data))
image = copy.deepcopy(distill_data)
image = (image - image.min()) / (image.max() - image.min())
writer.add_images('train/Image', image)
writer.add_image('train/Image', image[6])


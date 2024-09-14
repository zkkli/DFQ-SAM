import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns
import numpy as np

act_list = torch.load('pth_save/act_list.pth')
act_list2 = torch.load('pth_save/qkv_list.pth')
# act_list3 = torch.load('out_list_L_joke2.pth')



# violin plot
colorL = ['red', 'blue']
for i, (act1, act2) in enumerate(zip(act_list, act_list2)):
    # print(len(act2))
    print(i, act1.shape, act2[0].shape)  # 0 torch.Size([1, 16, 16, 768]) torch.Size([1, 16, 16, 768])
    act2 = act2[0]
    # exit()
    plt.figure(figsize=[20,10])
    ax = plt.subplot()
    # act1 =hooks_b[itr_hook].feature
    # act_list.append(act1)
    _, H, W, C = act1.shape
    m, n =300, 310
    
    plt.style.use('ggplot')
    
    act1 = act1.view(-1, C)
    act2 = act2.view(-1, C)
    # act3 = act3.view(-1, C)
    # act1_np = act1[:,300:350].cpu().numpy()
    act1_np = act1[:,m:n].cpu().numpy()
    act2_np = act2[:,m:n].cpu().numpy()
    # act3_np = act3[:,m:n].cpu().numpy()
    # act2_np = act2_np-act3_np
    # act1_np = act1_np-act2_np
    labels1 = np.array([str(i) for i in range(m,n)]*act1.shape[0])
    labels2 = np.array([str(i) for i in range(m,n)]*act2.shape[0])
    # print(labels.shape)
    # labels = labels+labels
    df1 = pd.DataFrame({'Category': labels1.flatten(), 'Values': act1_np.flatten()})
    df2 = pd.DataFrame({'Category': labels2.flatten(), 'Values': act2_np.flatten()})
    # bplot1 = ax.boxplot(act1_np, sym='.',widths=1, patch_artist=True, showfliers=False, medianprops={'color':'gray'})
    # for box in bplot1['boxes']:
    #     box.set(facecolor='lightgray')
    # bplot2 = ax.boxplot(act2_np, sym='.',widths=0.6, patch_artist=True, showfliers=True, medianprops={'color':'blue'})
    # for box in bplot2['boxes']:
    #     box.set(facecolor='lightblue')
    # bplot3 = ax.boxplot(act3_np, sym='.',widths=0.3, patch_artist=True, showfliers=False, medianprops={'color':'pink'})
    # for box in bplot3['boxes']:
    #     box.set(facecolor='blue')
    sns.violinplot(x="Category", y="Values", data=df1, width=1, color='blue',alpha=0.5)
    sns.violinplot(x="Category", y="Values", data=df2, width=1, color='green',alpha=0.5)
    # ax.set_xticklabels(labels, rotation=90, fontsize=10)
    
        
    plt.savefig('./show_act_violin_s_'+str(i)+'.png')
    plt.close()
    
# his plot
# for i, (act1, act2) in enumerate(zip(act_list, act_list2)):
#     plt.figure(figsize=(6, 3))
#     act2 = act2[0]
#     act1 = act1.view(-1).cpu().numpy()
#     act2 = act2.view(-1).cpu().numpy()
#     s1 = pd.Series(act1)
#     s2 = pd.Series(act2)
#     s1.hist(bins = 30,histtype = 'bar',align = 'mid',orientation = 'vertical',color='blue', alpha = 0.5,density = False, stacked=True)
#     s2.hist(bins = 30,histtype = 'bar',align = 'mid',orientation = 'vertical',color='green',alpha = 0.5,density = False, stacked=True)
#     plt.savefig('./show_act_his_s_'+str(i)+'.png')
#     plt.close()

# box plot
# colorL = ['red', 'blue']
# for i, (act1, act2) in enumerate(zip(act_list, act_list2)):
#     # print(len(act2))
#     print(i, act1.shape, act2[0].shape)  # 0 torch.Size([1, 16, 16, 768]) torch.Size([1, 16, 16, 768])
#     act2 = act2[0]
#     # exit()
#     plt.figure(figsize=[40,10])
#     ax = plt.subplot()
#     # act1 =hooks_b[itr_hook].feature
#     # act_list.append(act1)
#     _, H, W, C = act1.shape
#     m, n =300, 400
    
#     act1 = act1.view(-1, C)
#     act2 = act2.view(-1, C)
#     # act3 = act3.view(-1, C)
#     # act1_np = act1[:,300:350].cpu().numpy()
#     act1_np = act1[:,m:n].cpu().numpy()
#     act2_np = act2[:,m:n].cpu().numpy()
#     # act3_np = act3[:,m:n].cpu().numpy()
#     # act2_np = act2_np-act3_np
#     # act1_np = act1_np-act2_np
#     labels = [str(i) for i in range(m,n)]
#     labels = labels+labels
#     bplot1 = ax.boxplot(act1_np, sym='.',widths=1, patch_artist=True, showfliers=False, medianprops={'color':'gray'})
#     for box in bplot1['boxes']:
#         box.set(facecolor='lightgray')
#     bplot2 = ax.boxplot(act2_np, sym='.',widths=0.6, patch_artist=True, showfliers=True, medianprops={'color':'blue'})
#     for box in bplot2['boxes']:
#         box.set(facecolor='lightblue')
#     # bplot3 = ax.boxplot(act3_np, sym='.',widths=0.3, patch_artist=True, showfliers=False, medianprops={'color':'pink'})
#     # for box in bplot3['boxes']:
#     #     box.set(facecolor='blue')
#     ax.set_xticklabels(labels, rotation=90, fontsize=10)
    
        
#     plt.savefig('./show_act_box_s_'+str(i)+'.png')
#     plt.close()
    
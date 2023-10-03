#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load relevant packages
import datetime

import matplotlib.pyplot as plt
from esm import Alphabet, pretrained
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy.stats as ss
from prettytable import PrettyTable

from scipy.special import softmax

# # Use these packages if getting sequence from pdb website
# import requests as r
# from Bio import SeqIO
# from io import StringIO
    


# In[2]:


# Save sequence
sequence = 'MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRCALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL' 
data = [("Sars_Cov_2", sequence)]


# In[3]:


# Save 1v model names
models = ['esm1v_t33_650M_UR90S_'+str(i) for i in range(1,6) ]


# In[4]:


# Save relevant dataframe, remove extra rows and columns
t4_df = pd.read_csv("data_and_dms/T4_mutant_full_network_summary.csv",header=1)
t4_df = t4_df[~t4_df.variant.isnull()]
t4_df = t4_df.sort_values(["effect","pos","variant"])
t4_df.reset_index(inplace = True, drop = True)
t4_df = t4_df.drop("Unnamed: 0", axis = 1)
t4_df


# In[5]:


# set location to store models
torch.hub.set_dir("/gscratch/scrubbed/uwagura/models") # modify location for different machines

# Loop over the models
for mod in models: 
    
    # Load model and check if GPU is available
    model, alphabet = pretrained.load_model_and_alphabet(mod)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
    
    # Load batch converters
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    # Conduct DMS study for the given model
    dms = pd.DataFrame()
    for i in tqdm(range(batch_tokens.size(1))):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(
                #model(batch_tokens_masked)["logits"], dim=-1 # if gpu not available, use this
                model(batch_tokens_masked.cuda())["logits"], dim=-1
            )
        col = pd.DataFrame(token_probs[:, i].cpu())
        col = col.transpose()
        dms = pd.concat([dms,col],axis=1)
        
    # Relabel indices
    idx_map = {0: '<cls>', 1: '<pad>', 2: '<eos>', 3: '<unk>', 4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T', 12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y', 20: 'M', 21: 'H', 22: 'W', 23: 'C', 24: 'X', 25: 'B', 26: 'U', 27: 'Z', 28: 'O', 29: '.', 30: '-', 31: '<null_1>', 32: '<mask>'}
    dms = dms.rename(index = idx_map)
    
    # Relabel Columns
    new_col = list(sequence)
    new_col.append('<eos>')
    new_col.insert(0,'<cls>')
    dms.columns = new_col
    
    # Delete extraneous predictions and rare amino acids from DMS dataframe. Delete eos and cls token form sequence
    dms = dms.drop(index = ['<cls>','<pad>','<unk>','.','-','<null_1>','<mask>','X','B','U','Z','O'])
    dms = dms.drop(columns = ['<cls>','<eos>']) # DMS now has dimensions L x 20
    
    # Save DMS matrix
    dms.to_csv("dms/dms_cov_esm_"+str(models.index(mod))+".csv") # less detailed named
    #dms.to_csv("dms_cov_esm_"+str(models.index(mod))+"_on_"+str(datetime.datetime.now()).split(".")[0].replace(":","_").replace(" ","_at_")+".csv") # For more detailed name
    
    # Isolate the Pseudoenergies of the relevant mutations
    t4_df["logP_mut_"+mod] = ""
    t4_df["logP_wt_"+mod] = ""
    t4_df["log_mut_minus_wt_vals_"+mod] = ""
    for i, row in t4_df.iterrows():
        wt, idx, mt = row["wt"], int(row["pos"]), row["mut"]
        assert dms.columns.values.tolist()[idx-1] == wt,  "Wildtype at index "+str(i)+" does not match provided mutation"
        mut_pseudo_e = float(pd.DataFrame(dms.iloc[:, idx-1]).loc[mt if mt != '*' else '<eos>'])
        wt_pseudo_e = float(pd.DataFrame(dms.iloc[:, idx-1]).loc[wt])
        t4_df.loc[i,"logP_mut_"+mod] = mut_pseudo_e 
        t4_df.loc[i,"logP_wt_"+mod]  = wt_pseudo_e
        t4_df.loc[i,"log_mut_minus_wt_vals_"+mod] = float(mut_pseudo_e - wt_pseudo_e)
    
    print("Finished running "+mod)


# In[4]:


# Save dataframe with DMS study to a csv file
t4_df.to_csv("data_and_dms/t4_df_dms.csv")


# In[7]:


t4_df


# In[9]:


# Read T4 Lysozme data with results of last dms study
t4_df = pd.read_csv("data_and_dms/t4_df_dms.csv")


# In[10]:


# Plot prob vector from average pseudo energy at each site: 
fig,ax = plt.subplots(1,2,figsize=(10,10),dpi=300,gridspec_kw={'width_ratios': [1, 1]})

# Method 1: Take the average of the (logP_mut - logP_wt) values from each model
method_1 = []
for i,row in t4_df.iterrows():
    method_1.append( ((row['log_mut_minus_wt_vals_esm1v_t33_650M_UR90S_1'])+
                                   (row['log_mut_minus_wt_vals_esm1v_t33_650M_UR90S_2'])+
                                   (row['log_mut_minus_wt_vals_esm1v_t33_650M_UR90S_3'])+
                                   (row['log_mut_minus_wt_vals_esm1v_t33_650M_UR90S_4'])+
                                   (row['log_mut_minus_wt_vals_esm1v_t33_650M_UR90S_5']))/5
                        ) 

method_1= np.array(method_1)

# Plot the vector
method_1 = method_1.reshape(40,1) # note that the imshow function expects a vector with more than one dimension
im1 = ax[0].imshow(method_1,cmap='RdBu')
ax[0].set_title(r'$\frac{1}{5} \sum(\logP_{mut} - \logP_{wt}) $')
ax[0].set_xticks([])
ax[0].set_yticks(range(len(t4_df[[x for x in t4_df.columns if 'mut_struct_prob_diff' in x]])))
ax[0].set_yticklabels(labels=zip(t4_df['variant'].to_numpy(),t4_df['effect'].to_numpy()))
fig.colorbar(im1, ax=ax[0],orientation='vertical', aspect=1, pad=0.1)
me_sum = method_1

# Method 2: Take the log of the averaage P_mut, subtract the log of the average P_wt 
method_2 = []
for i,row in t4_df.iterrows():
    method_2.append( np.log( (np.exp(row['logP_mut_esm1v_t33_650M_UR90S_1'])+
                                np.exp(row['logP_mut_esm1v_t33_650M_UR90S_2'])+
                                np.exp(row['logP_mut_esm1v_t33_650M_UR90S_3'])+
                                np.exp(row['logP_mut_esm1v_t33_650M_UR90S_4'])+
                                np.exp(row['logP_mut_esm1v_t33_650M_UR90S_5']))
                            / 
                            (np.exp(row['logP_wt_esm1v_t33_650M_UR90S_1'])+
                                    np.exp(row['logP_wt_esm1v_t33_650M_UR90S_2'])+
                                    np.exp(row['logP_wt_esm1v_t33_650M_UR90S_3'])+
                                    np.exp(row['logP_wt_esm1v_t33_650M_UR90S_4'])+
                                    np.exp(row['logP_wt_esm1v_t33_650M_UR90S_5']))
                                 )
                         )


method_2= np.array(method_2)

# Plot the second vector
method_2 = method_2.reshape(40,1)
im1 = ax[1].imshow(method_2,cmap='RdBu')
ax[1].set_title(r'$ \log(\frac{1}{5}\sum P_{mut}) - \log(\frac{1}{5}\sum P_{wt}) $')
ax[1].set_xticks([])
ax[1].set_yticks(range(len(t4_df[[x for x in t4_df.columns if 'mut_struct_prob_diff' in x]])))
ax[1].set_yticklabels(labels=zip(t4_df['variant'].to_numpy(),t4_df['effect'].to_numpy()))
fig.colorbar(im1, ax=ax[1],orientation='vertical', aspect=1, pad=0.1)
plt.savefig('T4_mutant_log_differences_wt_structures.pdf')
plt.show()
me_log = method_2


# In[36]:


# Add combined results to T4_df
t4_df["Avg_delta_log"] = method_1
t4_df["Diff_log_sum_probs"] = method_2

# Save combine dataframe
t4_df.to_csv("t4_df_with_averages.csv")


# In[11]:


# Read T4 dataframe with averages from two methods above
t4_df = pd.read_csv("data_and_dms/t4_df_with_averages.csv")


# In[28]:


# Calculate Correlations between 
t4df_no_nan = t4_df[~np.isnan(t4_df["ddG"])]
delta_delta_G = t4df_no_nan["ddG"]
method_1_no_nan = t4df_no_nan["Avg_delta_log"]
method_2_no_nan = t4df_no_nan["Diff_log_sum_probs"]

avg_delt_log_rho,avg_delt_log_p = ss.spearmanr(delta_delta_G,method_1_no_nan)
diff_log_sum_rho,diff_log_sum_p = ss.spearmanr(delta_delta_G,method_1_no_nan)

# Plot correlations
fig,ax = plt.subplots(1,2,figsize=(5,2.5),dpi=300,gridspec_kw={'width_ratios': [1, 1]})

# Get minimum values of each array for plotting vertical and horizontal lines
min_x  = min(method_1_no_nan)
max_x = max(method_1_no_nan)
min_y  = min(delta_delta_G)
max_y = max(delta_delta_G)

# Method 1
ax[0].plot(method_1_no_nan, delta_delta_G, 'bo')
ax[0].set_title("Method 1")
ax[0].hlines(0, min_x, max_x, colors = '00', linestyles = "dashed" )
ax[0].vlines(0, min_y, max_y, colors = '00', linestyles = "dashed" )
ax[0].tick_params(labelsize = 9)
ax[0].set_xlabel(r"Network Prediction, $\frac{1}{5} \sum (logP_{mut} -  logP_{wt})$",{'fontsize': 'xx-small'})
ax[0].set_ylabel(r"$\Delta \Delta G$",labelpad = -7)

# Method 2
ax[1].plot(method_2_no_nan, delta_delta_G, 'ro')
ax[1].set_title("Method 2")
ax[1].hlines(0, min_x, max_x, colors = '00', linestyles = "dashed" )
ax[1].vlines(0, min_y, max_y, colors = '00', linestyles = "dashed" )
ax[1].tick_params(labelsize = 9)
ax[1].set_xlabel(r"Network Prediction, $log(\frac{1}{5} \sum P_{mut}) - log(\frac{1}{5} \sum P_{wt})$",{'fontsize': 'xx-small'})
ax[1].set_ylabel(r"$\Delta \Delta G$",labelpad = -7)
#plt.show()

# Display Correlations as table
correlations = PrettyTable(["Method", "Spearman R", "P-Value"])
correlations.add_row(["1", avg_delt_log_rho, avg_delt_log_p])
correlations.add_row(["2", diff_log_sum_rho, diff_log_sum_p])

plt.savefig("plots/ddG_vs_prediction.jpg")
print(correlations)
# print("For the first method ( taking the average of Delta Log values), the Spearman R value is: ", avg_delt_log_rho, " and the p-value is: ", avg_delt_log_p)
# print("For the second method ( taking the difference of the log sums), the Spearman R value is: ", diff_log_sum_rho, " and the p-value is: ", diff_log_sum_p)


# In[13]:


# Save pseudo energies as numpy arrays
method_1 = t4_df["Avg_delta_log"].to_numpy()
method_2 = t4_df["Diff_log_sum_probs"].to_numpy()


# In[27]:


# Plot ROC curves

# Styling for AUC text on plot
font = {'family': 'serif',
        'color':  'black',
        'size': 10
        }

box = {'facecolor': 'none',
       'edgecolor': 'green',
       'boxstyle': 'round'
      }

# Method one
num_mutations = method_1.size
num_n = 0
num_d = 0
total_n = t4_df[t4_df["effect"] == 'Neutral'].shape[0]
total_d = 40 - total_n
points_x = [0]
points_y = [0]
auc_1 = 0

for i, row in t4_df.sort_values("Avg_delta_log").iterrows():
    if row["effect"] == 'Destabilizing':
        num_d += 1
    else:
        num_n += 1
        auc_1 += (num_d/total_d)*(1/total_n)
        
    points_y.append(num_d/total_d )
    points_x.append( num_n/total_n )

fig,ax = plt.subplots(1,2,figsize=(5,2.5),dpi=300,gridspec_kw={'width_ratios': [1, 1]})
ax[0].set_xlim(-0.1,1.1)
ax[0].set_ylim(-0.1,1.1)
ax[0].tick_params(labelsize = 9)
ax[0].text(.50,.1,"AUC: "+str(round(auc_1,2)),fontdict = font,bbox=box)
ax[0].plot(points_x,points_y)
ax[0].plot((0,.5,1),(0,.5,1),linestyle = 'dotted',linewidth = 0.5,color = 'black') # Add diaganol line for random guess ROC curve

num_n = 0
num_d = 0
points_x = [0]
points_y = [0]
auc_2 = 0

for i, row in t4_df.sort_values("Diff_log_sum_probs").iterrows():
    if row["effect"] == 'Destabilizing':
        num_d += 1
        
    else:
        num_n += 1
        auc_2 += (num_d/total_d)*(1/total_n)
        
    points_y.append(num_d/total_d )
    points_x.append( num_n/total_n )

ax[1].set_xlim(-0.1,1.1)
ax[1].set_ylim(-0.1,1.1)
ax[1].tick_params(labelsize = 9)
ax[1].plot(points_x,points_y)
ax[1].plot((0,.5,1),(0,.5,1),linestyle = 'dotted',linewidth = 0.5,color = 'black') # Add diaganol line for random guess ROC curve
ax[1].text(.50,.1,"AUC: "+str(round(auc_2,3)),fontdict = font,bbox=box)
plt.savefig("plots/T4Lys_roc_curves.jpg")
#plt.show()


# In[ ]:





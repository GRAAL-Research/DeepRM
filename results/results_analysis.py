
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ast

with open('Test_4.txt', 'r') as det_:
    det = [line.strip().split('\t') for line in det_]
det_.close()
det = np.array(det)
df = pd.DataFrame(det[1:], columns=det[0])
df = df.astype({'train_acc': float,
                'valid_acc': float,
                'test_acc': float})

df_2 = df.groupby(by=['k', 'modl_1_dim', 'start_lr']).mean(numeric_only=True)
df_2 = df_2.reset_index()
df_2 = df_2.groupby(by=['k', 'modl_1_dim']).max(numeric_only=True)
df_2 = df_2.reset_index()
arr = np.array(df_2[['k', 'modl_1_dim', 'valid_acc']])
arr = np.array(df_2[['modl_1_dim']])
lis = []
for i in range(len(arr)):
    lis.append(ast.literal_eval(arr[i,0])[-1])
arr = np.vstack((np.array(df_2['k'], dtype=int), lis, np.array(df_2['valid_acc'])))
mea_arr = np.zeros((6,6))+1
for i in range(28):
    mea_arr[int(arr[0,i]/2),int(arr[1,i]/2)] = arr[2,i]

df_2 = df.groupby(by=['k', 'modl_1_dim', 'start_lr']).std(numeric_only=True)
df_2 = df_2.reset_index()
df_2 = df_2.groupby(by=['k', 'modl_1_dim']).max(numeric_only=True)
df_2 = df_2.reset_index()
arr = np.array(df_2[['k', 'modl_1_dim', 'valid_acc']])
arr = np.array(df_2[['modl_1_dim']])
lis = []
for i in range(len(arr)):
    lis.append(ast.literal_eval(arr[i,0])[-1])
arr = np.vstack((np.array(df_2['k'], dtype=int), lis, np.array(df_2['valid_acc'])))
std_arr = np.zeros((6,6))+1
for i in range(28):
    std_arr[int(arr[0,i]/2),int(arr[1,i]/2)] = arr[2,i]

plt.imshow(np.flip(mea_arr,0), cmap='hot', interpolation='nearest', extent=[0,10,0,10])
plt.colorbar()
plt.title('Mean acuracy as a function of compression size, message size')
plt.xlabel('Message size')
plt.ylabel('Sample compression sizeMessage size')
plt.show()

plt.imshow(np.flip(std_arr,0), cmap='hot', interpolation='nearest', extent=[0,10,0,10])
plt.colorbar()
plt.title('Mean std as a function of compression size, message size')
plt.xlabel('Message size')
plt.ylabel('Sample compression size')
plt.show()
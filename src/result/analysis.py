from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("/home/leblancbe/Downloads/wandb_export_2024-08-22T09_14_42.796-04_00.csv")

df = df[['msg_type', 'msg_size', 'compression_set_size', 'train_acc', 'valid_acc', 'test_acc', 'bound_mrch', 'bound_kl']]
df = df.sort_values(by=['msg_type', 'msg_size', 'compression_set_size', 'train_acc', 'valid_acc', 'test_acc', 'bound_mrch', 'bound_kl'])

df = df.astype({'msg_size': float})
df = df.astype({'compression_set_size': float})
df = df.astype({'train_acc': float})
df = df.astype({'valid_acc': float})
df = df.astype({'test_acc': float})
df = df.astype({'bound_mrch': float})
df = df.astype({'bound_kl': float})

df = df.loc[df['compression_set_size'] <= 15]
df_grouped = df.groupby(by=['msg_size', 'msg_type', 'compression_set_size']).max().reset_index()
df_merged = df_grouped[['msg_type', 'msg_size', 'compression_set_size', 'valid_acc']].merge(
    df[['msg_type', 'msg_size', 'compression_set_size', 'train_acc', 'valid_acc', 'test_acc', 'bound_mrch', 'bound_kl']],
    left_on=['msg_type', 'msg_size', 'compression_set_size', 'valid_acc'],
    right_on=['msg_type', 'msg_size', 'compression_set_size', 'valid_acc'])

df_merged = df_merged.sort_values(by=['msg_type', 'msg_size', 'compression_set_size']).reset_index()
#print(len(df_merged))
#for i in range(len(df_merged)) :
#    print(str(df_merged['msg_type'][i]) + \
#          ' & ' + str(int(df_merged['msg_size'][i])) + \
#          ' & ' + str(int(df_merged['compression_set_size'][i])) + \
#          ' & ' + str(round(df_merged['test_acc'][i] * 100,2)) +"\\\\")

print("\nMsg_type == cnt, metric == test_acc")
for i in range(len(df_merged) // 16) :
    print(str(round(df_merged['test_acc'][8*i] * 100,2)) + \
          ' & ' + str(round(df_merged['test_acc'][8 * i + 1] * 100,2)) +
          ' & ' + str(round(df_merged['test_acc'][8 * i + 2] * 100, 2)) +
          ' & ' + str(round(df_merged['test_acc'][8 * i + 3] * 100, 2)) +
          ' & ' + str(round(df_merged['test_acc'][8 * i + 4] * 100, 2)) +
          ' & ' + str(round(df_merged['test_acc'][8 * i + 5] * 100, 2)) +
          ' & ' + str(round(df_merged['test_acc'][8 * i + 6] * 100, 2)) +
          ' & ' + str(round(df_merged['test_acc'][8*i + 7] * 100,2)) +"\\\\")

print("\nMsg_type == dsc, metric == test_acc")
for i in range(len(df_merged) // 16, len(df_merged) // 8) :
    print(str(round(df_merged['test_acc'][8*i] * 100,2)) + \
          ' & ' + str(round(df_merged['test_acc'][8 * i + 1] * 100,2)) +
          ' & ' + str(round(df_merged['test_acc'][8 * i + 2] * 100, 2)) +
          ' & ' + str(round(df_merged['test_acc'][8 * i + 3] * 100, 2)) +
          ' & ' + str(round(df_merged['test_acc'][8 * i + 4] * 100, 2)) +
          ' & ' + str(round(df_merged['test_acc'][8 * i + 5] * 100, 2)) +
          ' & ' + str(round(df_merged['test_acc'][8 * i + 6] * 100, 2)) +
          ' & ' + str(round(df_merged['test_acc'][8 * i + 7] * 100,2)) +"\\\\")

print("\nMsg_type == cnt, metric == bound_kl")
for i in range(len(df_merged) // 16) :
    print(str(round(df_merged['bound_kl'][8*i] * 100,2)) + \
          ' & ' + str(round(df_merged['bound_kl'][8 * i + 1] * 100,2)) +
          ' & ' + str(round(df_merged['bound_kl'][8 * i + 2] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_kl'][8 * i + 3] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_kl'][8 * i + 4] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_kl'][8 * i + 5] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_kl'][8 * i + 6] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_kl'][8*i + 7] * 100,2)) +"\\\\")

print("\nMsg_type == dsc, metric == bound_kl")
for i in range(len(df_merged) // 16, len(df_merged) // 8) :
    print(str(round(df_merged['bound_kl'][8*i] * 100,2)) + \
          ' & ' + str(round(df_merged['bound_kl'][8 * i + 1] * 100,2)) +
          ' & ' + str(round(df_merged['bound_kl'][8 * i + 2] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_kl'][8 * i + 3] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_kl'][8 * i + 4] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_kl'][8 * i + 5] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_kl'][8 * i + 6] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_kl'][8 * i + 7] * 100,2)) +"\\\\")

print("\nMsg_type == cnt, metric == bound_mrch")
for i in range(len(df_merged) // 16) :
    print(str(round(df_merged['bound_mrch'][8*i] * 100,2)) + \
          ' & ' + str(round(df_merged['bound_mrch'][8 * i + 1] * 100,2)) +
          ' & ' + str(round(df_merged['bound_mrch'][8 * i + 2] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_mrch'][8 * i + 3] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_mrch'][8 * i + 4] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_mrch'][8 * i + 5] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_mrch'][8 * i + 6] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_mrch'][8*i + 7] * 100,2)) +"\\\\")

print("\nMsg_type == dsc, metric == bound_mrch")
for i in range(len(df_merged) // 16, len(df_merged) // 8) :
    print(str(round(df_merged['bound_mrch'][8*i] * 100,2)) + \
          ' & ' + str(round(df_merged['bound_mrch'][8 * i + 1] * 100,2)) +
          ' & ' + str(round(df_merged['bound_mrch'][8 * i + 2] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_mrch'][8 * i + 3] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_mrch'][8 * i + 4] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_mrch'][8 * i + 5] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_mrch'][8 * i + 6] * 100, 2)) +
          ' & ' + str(round(df_merged['bound_mrch'][8 * i + 7] * 100,2)) +"\\\\")


### Test_acc, cnt ###
rslts_array = np.zeros((8, 11))
for i in range(len(df_merged) // 16) :
    rslts_array[0, i] = str(round(df_merged['test_acc'][8*i] * 100,2))
    rslts_array[1, i] = str(round(df_merged['test_acc'][8 * i + 1] * 100,2))
    rslts_array[2, i] = str(round(df_merged['test_acc'][8 * i + 2] * 100, 2))
    rslts_array[3, i] = str(round(df_merged['test_acc'][8 * i + 3] * 100, 2))
    rslts_array[4, i] = str(round(df_merged['test_acc'][8 * i + 4] * 100, 2))
    rslts_array[5, i] = str(round(df_merged['test_acc'][8 * i + 5] * 100, 2))
    rslts_array[6, i] = str(round(df_merged['test_acc'][8 * i + 6] * 100, 2))
    rslts_array[7, i] = str(round(df_merged['test_acc'][8*i + 7] * 100,2))

fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(rslts_array, cmap="Greys")
x_axis_labels = [1, 2, 4, 6, 8, 10, 15, 20, 25, 50, 100]
y_axis_labels = [0, 1, 2, 4, 6, 8, 10, 15]

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(x_axis_labels)), labels=x_axis_labels)
ax.set_yticks(np.arange(len(y_axis_labels)), labels=y_axis_labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
mid = (np.max(rslts_array) + np.min(rslts_array)) / 2
for i in range(len(x_axis_labels)):
    for j in range(len(y_axis_labels)):
        col = "w" if rslts_array[j, i] > mid else "black"
        text = ax.text(i, j, rslts_array[j, i],
                       ha="center", va="center", color=col)

ax.set_title("Test accuracy on multiclass-MNIST (continuous messages)")
ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
plt.xlabel("Message size")
plt.ylabel("Compression set size")
cbar = ax.figure.colorbar(im, ax=ax)
fig.tight_layout()
plt.savefig("test_acc-continuous.png")
plt.show()



### Test_acc, dsc ###
rslts_array = np.zeros((8, 12))
for i in range(len(df_merged) // 16, len(df_merged) // 8) :
    rslts_array[0, i - len(df_merged) // 16] = str(round(df_merged['test_acc'][8*i] * 100,2))
    rslts_array[1, i - len(df_merged) // 16] = str(round(df_merged['test_acc'][8 * i + 1] * 100,2))
    rslts_array[2, i - len(df_merged) // 16] = str(round(df_merged['test_acc'][8 * i + 2] * 100, 2))
    rslts_array[3, i - len(df_merged) // 16] = str(round(df_merged['test_acc'][8 * i + 3] * 100, 2))
    rslts_array[4, i - len(df_merged) // 16] = str(round(df_merged['test_acc'][8 * i + 4] * 100, 2))
    rslts_array[5, i - len(df_merged) // 16] = str(round(df_merged['test_acc'][8 * i + 5] * 100, 2))
    rslts_array[6, i - len(df_merged) // 16] = str(round(df_merged['test_acc'][8 * i + 6] * 100, 2))
    rslts_array[7, i - len(df_merged) // 16] = str(round(df_merged['test_acc'][8*i + 7] * 100,2))
rslts_array[0,0] = np.min(rslts_array)

fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(rslts_array, cmap="Greys")
x_axis_labels = [0, 1, 2, 4, 6, 8, 10, 15, 20, 25, 50, 100]
y_axis_labels = [0, 1, 2, 4, 6, 8, 10, 15]

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(x_axis_labels)), labels=x_axis_labels)
ax.set_yticks(np.arange(len(y_axis_labels)), labels=y_axis_labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
mid = (np.max(rslts_array) + np.min(rslts_array)) / 2
for i in range(len(x_axis_labels)):
    for j in range(len(y_axis_labels)):
        col = "w" if rslts_array[j, i] > mid else "black"
        texx = rslts_array[j, i] if i + j > 0 else "---"
        text = ax.text(i, j, texx,
                       ha="center", va="center", color=col)

ax.set_title("Test accuracy on multiclass-MNIST (discrete messages)")
ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
plt.xlabel("Message size")
plt.ylabel("Compression set size")
cbar = ax.figure.colorbar(im, ax=ax)
fig.tight_layout()
plt.savefig("test_acc-discrete.png")
plt.show()



### kl bound, cnt ###
rslts_array = np.zeros((8, 11))
for i in range(len(df_merged) // 16) :
    rslts_array[0, i] = str(round(df_merged['bound_kl'][8*i] * 100,2))
    rslts_array[1, i] = str(round(df_merged['bound_kl'][8 * i + 1] * 100,2))
    rslts_array[2, i] = str(round(df_merged['bound_kl'][8 * i + 2] * 100, 2))
    rslts_array[3, i] = str(round(df_merged['bound_kl'][8 * i + 3] * 100, 2))
    rslts_array[4, i] = str(round(df_merged['bound_kl'][8 * i + 4] * 100, 2))
    rslts_array[5, i] = str(round(df_merged['bound_kl'][8 * i + 5] * 100, 2))
    rslts_array[6, i] = str(round(df_merged['bound_kl'][8 * i + 6] * 100, 2))
    rslts_array[7, i] = str(round(df_merged['bound_kl'][8*i + 7] * 100,2))

fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(rslts_array, cmap="Greys")
x_axis_labels = [1, 2, 4, 6, 8, 10, 15, 20, 25, 50, 100]
y_axis_labels = [0, 1, 2, 4, 6, 8, 10, 15]

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(x_axis_labels)), labels=x_axis_labels)
ax.set_yticks(np.arange(len(y_axis_labels)), labels=y_axis_labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
mid = (np.max(rslts_array) + np.min(rslts_array)) / 2
for i in range(len(x_axis_labels)):
    for j in range(len(y_axis_labels)):
        col = "w" if rslts_array[j, i] > mid else "black"
        text = ax.text(i, j, rslts_array[j, i],
                       ha="center", va="center", color=col)

ax.set_title("Generalization bound (on accuracy, KL) on multiclass-MNIST (continuous messages)")
ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
plt.xlabel("Message size")
plt.ylabel("Compression set size")
cbar = ax.figure.colorbar(im, ax=ax)
fig.tight_layout()
plt.savefig("kl-continuous.png")
plt.show()



### kl bound, dsc ###
rslts_array = np.zeros((8, 12))
for i in range(len(df_merged) // 16, len(df_merged) // 8) :
    rslts_array[0, i - len(df_merged) // 16] = str(round(df_merged['bound_kl'][8*i] * 100,2))
    rslts_array[1, i - len(df_merged) // 16] = str(round(df_merged['bound_kl'][8 * i + 1] * 100,2))
    rslts_array[2, i - len(df_merged) // 16] = str(round(df_merged['bound_kl'][8 * i + 2] * 100, 2))
    rslts_array[3, i - len(df_merged) // 16] = str(round(df_merged['bound_kl'][8 * i + 3] * 100, 2))
    rslts_array[4, i - len(df_merged) // 16] = str(round(df_merged['bound_kl'][8 * i + 4] * 100, 2))
    rslts_array[5, i - len(df_merged) // 16] = str(round(df_merged['bound_kl'][8 * i + 5] * 100, 2))
    rslts_array[6, i - len(df_merged) // 16] = str(round(df_merged['bound_kl'][8 * i + 6] * 100, 2))
    rslts_array[7, i - len(df_merged) // 16] = str(round(df_merged['bound_kl'][8*i + 7] * 100,2))
rslts_array[0,0] = np.min(rslts_array)

fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(rslts_array, cmap="Greys")
x_axis_labels = [0, 1, 2, 4, 6, 8, 10, 15, 20, 25, 50, 100]
y_axis_labels = [0, 1, 2, 4, 6, 8, 10, 15]

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(x_axis_labels)), labels=x_axis_labels)
ax.set_yticks(np.arange(len(y_axis_labels)), labels=y_axis_labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
mid = (np.max(rslts_array) + np.min(rslts_array)) / 2
for i in range(len(x_axis_labels)):
    for j in range(len(y_axis_labels)):
        col = "w" if rslts_array[j, i] > mid else "black"
        texx = rslts_array[j, i] if i + j > 0 else "---"
        text = ax.text(i, j, texx,
                       ha="center", va="center", color=col)

ax.set_title("Generalization bound (on accuracy, KL) on multiclass-MNIST (discrete messages)")
ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
plt.xlabel("Message size")
plt.ylabel("Compression set size")
cbar = ax.figure.colorbar(im, ax=ax)
fig.tight_layout()
plt.savefig("kl-discrete.png")
plt.show()



### mrch bound, cnt ###
rslts_array = np.zeros((8, 11))
for i in range(len(df_merged) // 16) :
    rslts_array[0, i] = str(round(df_merged['bound_mrch'][8*i] * 100,2))
    rslts_array[1, i] = str(round(df_merged['bound_mrch'][8 * i + 1] * 100,2))
    rslts_array[2, i] = str(round(df_merged['bound_mrch'][8 * i + 2] * 100, 2))
    rslts_array[3, i] = str(round(df_merged['bound_mrch'][8 * i + 3] * 100, 2))
    rslts_array[4, i] = str(round(df_merged['bound_mrch'][8 * i + 4] * 100, 2))
    rslts_array[5, i] = str(round(df_merged['bound_mrch'][8 * i + 5] * 100, 2))
    rslts_array[6, i] = str(round(df_merged['bound_mrch'][8 * i + 6] * 100, 2))
    rslts_array[7, i] = str(round(df_merged['bound_mrch'][8*i + 7] * 100,2))

fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(rslts_array, cmap="Greys")
x_axis_labels = [1, 2, 4, 6, 8, 10, 15, 20, 25, 50, 100]
y_axis_labels = [0, 1, 2, 4, 6, 8, 10, 15]

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(x_axis_labels)), labels=x_axis_labels)
ax.set_yticks(np.arange(len(y_axis_labels)), labels=y_axis_labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
mid = (np.max(rslts_array) + np.min(rslts_array)) / 2
for i in range(len(x_axis_labels)):
    for j in range(len(y_axis_labels)):
        col = "w" if rslts_array[j, i] > mid else "black"
        text = ax.text(i, j, rslts_array[j, i],
                       ha="center", va="center", color=col)

ax.set_title("Generalization bound (on accuracy, Marchand) on multiclass-MNIST (continuous messages)")
ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
plt.xlabel("Message size")
plt.ylabel("Compression set size")
cbar = ax.figure.colorbar(im, ax=ax)
fig.tight_layout()
plt.savefig("marchand-continuous.png")
plt.show()



### mrch bound, dsc ###
rslts_array = np.zeros((8, 12))
for i in range(len(df_merged) // 16, len(df_merged) // 8) :
    rslts_array[0, i - len(df_merged) // 16] = str(round(df_merged['bound_mrch'][8*i] * 100,2))
    rslts_array[1, i - len(df_merged) // 16] = str(round(df_merged['bound_mrch'][8 * i + 1] * 100,2))
    rslts_array[2, i - len(df_merged) // 16] = str(round(df_merged['bound_mrch'][8 * i + 2] * 100, 2))
    rslts_array[3, i - len(df_merged) // 16] = str(round(df_merged['bound_mrch'][8 * i + 3] * 100, 2))
    rslts_array[4, i - len(df_merged) // 16] = str(round(df_merged['bound_mrch'][8 * i + 4] * 100, 2))
    rslts_array[5, i - len(df_merged) // 16] = str(round(df_merged['bound_mrch'][8 * i + 5] * 100, 2))
    rslts_array[6, i - len(df_merged) // 16] = str(round(df_merged['bound_mrch'][8 * i + 6] * 100, 2))
    rslts_array[7, i - len(df_merged) // 16] = str(round(df_merged['bound_mrch'][8*i + 7] * 100,2))
rslts_array[0,0] = np.min(rslts_array)

fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(rslts_array, cmap="Greys")
x_axis_labels = [0, 1, 2, 4, 6, 8, 10, 15, 20, 25, 50, 100]
y_axis_labels = [0, 1, 2, 4, 6, 8, 10, 15]

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(x_axis_labels)), labels=x_axis_labels)
ax.set_yticks(np.arange(len(y_axis_labels)), labels=y_axis_labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
mid = (np.max(rslts_array) + np.min(rslts_array)) / 2
for i in range(len(x_axis_labels)):
    for j in range(len(y_axis_labels)):
        col = "w" if rslts_array[j, i] > mid else "black"
        texx = rslts_array[j, i] if i + j > 0 else "---"
        text = ax.text(i, j, texx,
                       ha="center", va="center", color=col)

ax.set_title("Generalization bound (on accuracy, Marchand) on multiclass-MNIST (discrete messages)")
ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
plt.xlabel("Message size")
plt.ylabel("Compression set size")
cbar = ax.figure.colorbar(im, ax=ax)
fig.tight_layout()
plt.savefig("marchand-discrete.png")
plt.show()
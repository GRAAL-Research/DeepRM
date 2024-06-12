import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ast

mea_arr =np.array([[0.5, 0.668, 0.739, 0.802, 0.994],
                   [0.775, 0.808, 0.870, 0.956, 0.962],
                   [0.804, 0.823, 0.889, 0.971, 0.974]])
mea_arr = np.flip(mea_arr, 0)
fig, ax = plt.subplots()
im = ax.imshow(mea_arr, cmap='hot')
#plt.imshow(np.flip(mea_arr, 0), cmap='hot', interpolation='nearest', extent=[0,10,0,10])
#plt.colorbar()
#plt.show()



# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(5), labels=[0,1,2,4,10])
ax.set_yticks(np.arange(3), labels=[10,6,0])

# Loop over data dimensions and create text annotations.
for i in range(3):
    for j in range(5):
        text = ax.text(j, i, mea_arr[i, j],
                       ha="center", va="center", color="black")
ax.set_title("Mean acuracy as a function of compression size, message size")
#plt.xlabel('Message size')
#plt.ylabel('Sample compression sizeMessage size')
fig.tight_layout()
#fig.colorbar()
plt.show()
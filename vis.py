from epinet_fun.func_pfm import read_pfm
import matplotlib.pyplot as plt
data_path = './epinet_output/2067.pfm' 
data = read_pfm(data_path)

plt.imshow(data[0,:,:,0])
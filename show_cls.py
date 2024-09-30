from csv_rw import *
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd 


path = 'results/l2svm_celebahq_with0_u1_s1_r10_i0_w0_gp150_im256_ss1000_ni30000_bs4_ug0.5/'
filename = path+'cls_acc_5.csv'
cls = read_csv(filename)
plt.plot(cls, label='cls_acc')
df_loss_file = path+'df_loss.csv'
data = pd.read_csv(df_loss_file)
df_loss = []
start = 1000
end = 84001
step = 1000
steps = [1014, 2036, 3038, 4036, 5106, 6200, 7114, 8066, 9260, 10276, 11020, 12
for i in range(start, end, step):
    df_loss.append(data['Value'][data])
plt.plot(df_loss, label='df_loss')
plt.savefig(path+'all_cls.png')

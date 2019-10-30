import glob
import os
import pickle
data_dir = "/tigress/LEIFER/Xinwei/github/Pytorch-Unet/output"
worm_list = sorted(glob.glob(os.path.join(data_dir, '*.txt')))

for file_name in worm_list:

    with open(file_name, "rb") as fp:
        try:
            cline_dict = pickle.load(fp)
            fp.close()
        except:
            os.system('rm ' + file_name)
            print(file_name)
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Utils import dataloader


def data_analysis():
    count = 0
    ixi_id = []
    scanner_name = []
    contrast_type = []
    file_path = []
    print('start of data analysis')
    t1_train_path,t2_train_path,file_names_train_t1,file_names_train_t2,file_names_val_t1,file_names_val_t2 = dataloader.DataLoader().data
    print(t1_train_path)
    for root, dirs, files in os.walk(t1_train_path):
        print(root)
        for file in files:
            parameters_search = re.search('IXI(\d*)-(\w*)-(\d*)-(\w*)\.(.*?)', file, re.IGNORECASE)
            if parameters_search:
                ixi_id.append(int(parameters_search.group(1)))
                scanner_name.append(parameters_search.group(2))
                contrast_type.append("T1")
                file_path.append(os.path.join(t1_train_path, file))
            count = count + 1
        print("processed T1 {} files".format(count))
    for root, dirs, files in os.walk(t2_train_path):
        for file in files:
            parameters_search = re.search('IXI(\d*)-(\w*)-(\d*)-(\w*)\.(.*?)', file, re.IGNORECASE)
            if parameters_search:
                ixi_id.append(int(parameters_search.group(1)))
                scanner_name.append(parameters_search.group(2))
                contrast_type.append("T2")
                file_path.append(os.path.join(t2_train_path, file))
            count = count + 1
        print("processed T2 {} files".format(count))
    plot_data(ixi_id,scanner_name,contrast_type,file_path)


def plot_data(ixi_id,scanner_name,contrast_type,file_path):
    out_df = pd.DataFrame(data={"our_ixi_id": ixi_id, "scanner_name": scanner_name, "contrast_type": contrast_type,
                                "file_path": file_path})
    sns.countplot(x="scanner_name", data=out_df)
    plt.show()

if __name__=='__main__':
    data_analysis()
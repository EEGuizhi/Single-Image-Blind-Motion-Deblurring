# **Datasets**

### **Folder Structure**
```
datasets/
├── realblur/
│   ├── RealBlur-J_ECC_IMCORR_centroid_itensity_ref/       (RealBlur-J dataset)
│   ├── RealBlur-R_BM3D_ECC_IMCORR_centroid_itensity_ref/  (RealBlur-R dataset)
│   ├── download_utils.py                                  (Used to download RealBlur datasets)
│   ├── RealBlur_J_test_list.txt                           (Test list in RealBlur-J dataset)
│   ├── RealBlur_J_test_list.json                          (Generated from preprocess.py)
│   ├── ...
│   ├── RealBlur_R_train_list.txt                          (Train list in RealBlur-J dataset)
│   └── RealBlur_R_train_list.json                         (Generated from preprocess.py)
├── augmentation.py  (Custom data augmentation for training)
└── dataset.py       (Custom dataset for training and testing)
```

### **Archive Introduction**
- `download_utils.py`：可以用來下載 RealBlur Dataset。

- `augmentation.py`：訓練時所使用的自訂義 Data Augmentation。

- `dataset.py`：訓練與測試時用來讀取 Input/Ground-truth 圖片用的自訂義 Dataset。

### **Notes**
- 如果 `realblur` 資料夾裡面沒有像上方描述的那些檔案，代表你還沒有下載 RealBlur Dataset，
  請使用 `download_utils.py` 將檔案下載到該處並解壓縮。

- 若缺少如 `RealBlur_J_test_list.json` 等 `.json` 檔案代表還沒有前處理過資料集，
  請使用根目錄的 `preprocess.py` 來生成 `dataset.py` 運作時所需的那些 `.json` 檔案。

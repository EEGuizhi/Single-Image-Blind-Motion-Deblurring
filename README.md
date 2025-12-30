# **Single-Image-Blind-Motion-Deblurring**
- **Project**: NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
- **Members**: BSChen (陳柏翔), JRKang (康鈞睿)

### **Tutorial**
1. 依照 `requirements.txt` 安裝所需套件。
2. 下載 RealBlur 資料集。 (請見 `./dataset/` 資料夾中的 README.md)
3. Preprocess (請見 `./dataset/` 資料夾中的 README.md)
4. 下載 Pretrained Model Weights 到 `./pretrain_weights` 資料夾中。
5. 可以執行了！

### **Introduction**
- `./configs/config.py`:
    所有超參數都在此檔案中進行設置。 (不論訓練/測試/推論都是從此檔案讀取 configs)
- `./train.py`:
    訓練模型使用的主執行檔。
- `./test.py`:
    測試模型使用的主執行檔。
- `./summary.py`:
    量測模型資訊的主執行檔。
- `./predict.py`:
    實際用來還原單張/多張清晰影像的主執行檔。

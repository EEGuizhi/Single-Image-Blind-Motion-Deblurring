# BSChen

### 2025/11/13 ###
- 完成 `download_utils.py`:
    可以在裡面定義要下載的 model 然後直接下載到此處。
- 完成 `show_modules.py`:
    可以 print 出某個 .pth 檔中的所有 modules name，方便判斷是哪個 model。
- 觀察:
    看起來 `realblur_j-width32.pth` 裡面的 model 是對應到 `MLWNet_arch.py` 而不是 `MIMO_arch.py`，
    因為裡面沒有 `MIMO_arch.py` 裡面的 SCM, FAM 等 modules。

### 


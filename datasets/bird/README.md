# 🐦 CUB-200-2011 Bird Image Retrieval System

使用 **CLIP + FAISS** 的鳥類圖片檢索系統，支援文字搜尋和圖片搜尋。

## 📁 專案結構

```
ImageRAG/
├── datasets/
│   └── bird/
│       ├── BirdRetrivealDataWithDescription.ipynb  # 建立 Index 的 Notebook (在 Colab 執行)
│       ├── search_bird.py                          # 搜尋工具 (本地或 Colab 使用)
│       ├── README.md                               # 本文件
│       ├── index/                                  # 已建立的 Index 檔案
│       │   ├── cub200_image.index                  # 純圖片 embedding
│       │   ├── cub200_text.index                   # 純文字 embedding
│       │   ├── cub200_metadata.pkl                 # 圖片路徑、類別等 metadata
│       │   └── path.json                           # 圖片路徑映射
│       └── CUB_200_2011/                           # 原始資料集 (需另外下載)
│           ├── images/                             # 11,788 張鳥類圖片
│           ├── classes.txt                         # 200 種鳥類類別
│           ├── image_class_labels.txt              # 圖片類別標籤
│           └── ...
└── ...
```

**注意**: 原始圖片資料集需另外下載 (見下方說明)

---

## 📥 Dataset 下載位置

### 方法 1: 從官方網站下載

1. 前往 [CUB-200-2011 Dataset](http://www.vision.caltech.edu/datasets/cub_200_2011/)
2. 下載 `CUB_200_2011.tgz`
3. 解壓縮到 `datasets/bird/` 目錄

### 方法 2: 使用命令列下載

```bash
# 下載
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

# 解壓縮
tar -xzf CUB_200_2011.tgz -C datasets/bird/
```

### 方法 3: 在 Colab 中下載

```python
!wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
!tar -xzf CUB_200_2011.tgz -C /content/data/
```

### 資料集結構

```
CUB_200_2011/
├── images/                     # 11,788 張圖片，200 個類別資料夾
│   ├── 001.Black_footed_Albatross/
│   ├── 002.Laysan_Albatross/
│   ├── ...
│   └── 200.Common_Yellowthroat/
├── classes.txt                 # 200 種鳥類類別名稱
├── images.txt                  # 圖片 ID 對應路徑
├── image_class_labels.txt      # 圖片對應類別
├── train_test_split.txt        # 訓練/測試集分割
├── bounding_boxes.txt          # 鳥類位置框
└── attributes/                 # 312 種視覺屬性標註
    └── image_attribute_labels.txt
```

### 文字描述資料 (cvpr2016_cub)

如需使用文字描述進行 embedding：

```bash
# 下載 Reed et al. 的文字描述
# 來源: https://github.com/reedscot/cvpr2016
```

```
cvpr2016_cub/
├── text_c10/           # 每張圖片 10 條文字描述
│   ├── 001.Black_footed_Albatross/
│   │   ├── Black_Footed_Albatross_0001_796111.txt
│   │   └── ...
│   └── ...
└── allclasses.txt      # 類別列表
```

---

## 🔧 建立 Index (使用 Colab)

### Step 1: 開啟 Notebook

在 Google Colab 中開啟 `BirdRetrivealDataWithDescription.ipynb`

### Step 2: 下載資料集

```python
!wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
!tar -xzf CUB_200_2011.tgz -C /content/data/
```

### Step 3: 載入 CLIP 模型

```python
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-bigG-14',
    pretrained='laion2b_s39b_b160k'
)
model = model.to("cuda")
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
```

### Step 4: 產生 Embedding

Notebook 會：
1. 讀取 200 種鳥類類別名稱
2. 為每張圖片產生 **Image Embedding**
3. 根據類別名稱或文字描述產生 **Text Embedding**
4. 結合兩者：`Combined = α × Image + (1-α) × Text`

### Step 5: 建立並儲存 FAISS Index

```python
import faiss

# 建立 Index
index_image = faiss.IndexFlatIP(image_array.shape[1])
index_image.add(image_array)

index_text = faiss.IndexFlatIP(text_array.shape[1])
index_text.add(text_array)

# 儲存
faiss.write_index(index_image, "cub200_image.index")
faiss.write_index(index_text, "cub200_text.index")
```

### Step 6: 下載 Index 檔案

```python
from google.colab import files

files.download('cub200_image.index')
files.download('cub200_text.index')
files.download('cub200_metadata.pkl')
files.download('path.json')
```

---

## 🔍 使用搜尋工具

### 安裝依賴

```bash
pip install faiss-cpu open_clip_torch torch numpy pillow
```

### 方法 1: 命令列使用

```bash
# 文字搜尋
python search_bird.py --index-dir ./index --query "yellow bird with black wings" --k 5

# 圖片搜尋
python search_bird.py --index-dir ./index --image "/path/to/bird.jpg" --k 5

# 指定 path.json
python search_bird.py --index-dir ./index --path-json ./index/path.json --query "cardinal"

# 互動模式
python search_bird.py --index-dir ./index --interactive
```

### 方法 2: 在 Python/Notebook 中使用

```python
from search_bird import BirdSearchEngine

# 初始化搜尋引擎 (模型只載入一次)
engine = BirdSearchEngine(
    index_dir="./index",
    path_json="./index/path.json"  # 可選
)

# 文字搜尋
results = engine.search_by_text("red cardinal bird", k=5)
engine.print_results(results)

# 圖片搜尋
results = engine.search_by_image("/path/to/query.jpg", k=5)
engine.print_results(results)

# 在 Notebook 中顯示圖片
engine.show_results_with_images(results)
```

### 方法 3: 互動模式

```bash
python search_bird.py --index-dir ./index --interactive
```

```
🎮 進入互動式搜尋模式
   輸入 'q' 或 'quit' 退出
   輸入 't:查詢文字' 進行文字搜尋
   輸入 'i:圖片路徑' 進行圖片搜尋

🔍 輸入查詢: blue jay
📝 文字搜尋: 'blue jay'

============================================================
🔍 搜尋結果:
============================================================

[1] Blue_Jay
    Score: 0.4521
    Description: This bird has blue and white feathers...

[2] Indigo_Bunting
    Score: 0.3892
...
```

---

## 📊 Index 類型說明

| Index 類型 | 說明 | 適用場景 |
|-----------|------|---------|
| `image` | 純圖片 embedding (預設) | 視覺相似度搜尋 |
| `text` | 純文字 embedding | 語義搜尋 |

```python
# 使用不同的 index
results = engine.search_by_text("yellow bird", k=5, index_type="image")
results = engine.search_by_text("yellow bird", k=5, index_type="text")
```

---

## ⚙️ 參數設定

### Embedding 權重 (Alpha)

在建立 Index 時設定：
```python
ALPHA = 0.6  # Image 權重 60%, Text 權重 40%
combined = ALPHA * img_emb + (1 - ALPHA) * text_emb
```

### 搜尋參數

| 參數 | 說明 | 預設值 |
|-----|------|-------|
| `--index-dir` | Index 目錄路徑 | `./index` |
| `--path-json` | path.json 路徑 | 自動尋找 |
| `--query` | 文字搜尋查詢 | - |
| `--image` | 圖片搜尋路徑 | - |
| `--k` | 返回結果數量 | 5 |
| `--index-type` | Index 類型 | `image` |

---

## 🚀 效能優化

### 模型快取

`search_bird.py` 使用全域模型快取，避免重複載入：

```python
# 第一次建立 - 載入模型 (約 60 秒)
engine1 = BirdSearchEngine(index_dir="./index")

# 第二次建立 - 使用快取 (瞬間完成)
engine2 = BirdSearchEngine(index_dir="./index")
```

### 預先載入模型

```python
from search_bird import get_clip_model

# 程式啟動時預先載入
get_clip_model()

# 之後使用都很快
engine = BirdSearchEngine(index_dir="./index")
```

---

## 📝 搜尋結果格式

```python
results = engine.search_by_text("cardinal", k=3)

# results 是 list of dict
[
    {
        "index": 1234,
        "path": "datasets/bird/CUB_200_2011/images/017.Cardinal/Cardinal_0001.jpg",
        "class_name": "Cardinal",
        "score": 0.4521,
        "description": "This bird has a red body with a pointed crest..."  # 如有
    },
    ...
]
```

---

## 🎯 範例查詢

| 查詢類型 | 範例 |
|---------|------|
| 顏色 | `"yellow bird"`, `"red and black bird"` |
| 鳥種 | `"cardinal"`, `"blue jay"`, `"sparrow"` |
| 特徵 | `"bird with long beak"`, `"bird with crest"` |
| 組合 | `"small yellow bird with black wings"` |

---

## 🔗 相關連結

- [CUB-200-2011 Dataset](http://www.vision.caltech.edu/datasets/cub_200_2011/)
- [Reed et al. Text Descriptions](https://github.com/reedscot/cvpr2016)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [FAISS](https://github.com/facebookresearch/faiss)

---

## 📄 License

This project uses the CUB-200-2011 Dataset for academic purposes.

Citation:
```bibtex
@techreport{WahCUB_200_2011,
    Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
    Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
    Year = {2011},
    Institution = {California Institute of Technology},
    Number = {CNS-TR-2011-001}
}
```

# 📁 ImageRAG Datasets

本目錄包含 ImageRAG 專案使用的資料集與 FAISS 檢索索引。

## 📊 支援的資料集

| 資料集 | 類別數 | 圖片數 | 說明 |
|-------|-------|-------|------|
| 🐦 [CUB-200-2011 (Bird)](./bird/) | 200 | 11,788 | 鳥類細粒度分類資料集 |
| 🚗 [Stanford Cars](./car/) | 196 | 16,185 | 汽車細粒度分類資料集 |

---

## 📁 目錄結構

```
datasets/
├── README.md                    # 本文件
├── bird/                        # 鳥類資料集
│   ├── README.md                # Bird 資料集說明
│   ├── BirdRetrivealDataWithDescription.ipynb  # 建立 Index 的 Notebook
│   ├── search_bird.py           # 搜尋工具
│   ├── index/                   # FAISS Index 檔案
│   │   ├── cub200_image.index
│   │   ├── cub200_text.index
│   │   ├── cub200_metadata.pkl
│   │   └── path.json
│   └── CUB_200_2011/            # 原始資料集 (需下載)
│       └── images/
└── car/                         # 汽車資料集
    ├── README.md                # Car 資料集說明
    ├── create_car_indexing.ipynb  # 建立 Index 的 Notebook
    ├── search_car.py            # 搜尋工具
    ├── index/                   # FAISS Index 檔案
    │   ├── cars_combined.index
    │   ├── cars_image.index
    │   ├── cars_text.index
    │   ├── cars_metadata.pkl
    │   └── path.json
    └── cars_train/              # 原始資料集 (需下載)
        └── cars_train/
```

---

## 🚀 快速開始

### 1. 下載原始資料集

#### Bird (CUB-200-2011)
```bash
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz -C datasets/bird/
```

#### Car (Stanford Cars)
```bash
# 使用 Kaggle API
kaggle datasets download -d eduardo4jesus/stanford-cars-dataset -p datasets/car/ --unzip
```

### 2. 建立 FAISS Index (在 Colab 執行)

1. 開啟對應的 Notebook：
   - Bird: `BirdRetrivealDataWithDescription.ipynb`
   - Car: `create_car_indexing.ipynb`
   
2. 執行所有 Cell，產生 embedding 並建立 index

3. 下載 `index/` 資料夾內的檔案

### 3. 使用搜尋功能

```python
from retrieval import init_faiss_retrieval, search_bird_image_path, search_car_image_path

# 初始化 (只需一次)
init_faiss_retrieval(
    bird_index_dir="datasets/bird/index",
    car_index_dir="datasets/car/index"
)

# 搜尋鳥類
bird_path = search_bird_image_path("yellow bird with black wings", k=1)
print(f"Bird: {bird_path}")

# 搜尋汽車
car_path = search_car_image_path("red sports car BMW", k=1)
print(f"Car: {car_path}")
```

---

## 🔧 技術細節

### CLIP 模型

Bird 和 Car 資料集都使用相同的 CLIP 模型：

```python
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-bigG-14',
    pretrained='laion2b_s39b_b160k'
)
tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
```

### Embedding 策略

| 資料集 | Alpha | 公式 |
|-------|-------|------|
| Bird | 0.6 | `0.6 × Image + 0.4 × Text` |
| Car | 0.7 | `0.7 × Image + 0.3 × Text` |

### Index 類型

| Index | 說明 | Bird | Car |
|-------|------|------|-----|
| `image.index` | 純圖片 embedding | ✅ | ✅ |
| `text.index` | 純文字 embedding | ✅ | ✅ |
| `combined.index` | Image + Text 結合 | ❌ | ✅ |

### path.json

`path.json` 儲存圖片的相對路徑，用於跨環境使用：

```json
[
  "datasets/bird/CUB_200_2011/images/001.Black_footed_Albatross/xxx.jpg",
  "datasets/car/cars_train/cars_train/00001.jpg",
  ...
]
```

---

## 📦 Index 檔案說明

### cub200_metadata.pkl / cars_metadata.pkl

```python
{
    "paths": [...],          # 圖片路徑列表
    "class_names": [...],    # 類別名稱列表
    "class_ids": [...],      # 類別 ID 列表 (Car)
    "classes": {...},        # 類別 ID 對應名稱 (Car)
    "descriptions": [...],   # 文字描述 (Bird)
    "alpha": 0.6             # Embedding 權重
}
```

---

## 🔗 資料集來源

| 資料集 | 官方網站 | 替代下載 |
|-------|---------|---------|
| CUB-200-2011 | [Caltech](http://www.vision.caltech.edu/datasets/cub_200_2011/) | [直接下載](https://data.caltech.edu/records/65de6-vp158) |
| Stanford Cars | [Stanford](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) | [Kaggle](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset) |

---

## 📄 License

這些資料集僅供學術研究使用，請遵循各資料集的授權條款。

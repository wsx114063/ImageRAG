# 🚗 Stanford Cars Image Retrieval System

使用 **CLIP + FAISS** 的汽車圖片檢索系統，支援文字搜尋和圖片搜尋。

## 📁 專案結構

```
ImageRAG/
├── datasets/
│   └── car/
│       ├── create_car_indexing.ipynb   # 建立 Index 的 Notebook (在 Colab 執行)
│       ├── search_car.py               # 搜尋工具 (本地或 Colab 使用)
│       ├── README.md                   # 本文件
│       └── index/                      # 已建立的 Index 檔案
│           ├── cars_combined.index     # 結合 image+text 的 embedding
│           ├── cars_image.index        # 純圖片 embedding
│           ├── cars_text.index         # 純文字 embedding
│           ├── cars_metadata.pkl       # 圖片路徑、類別等 metadata
│           └── path.json               # 圖片路徑映射
└── ...
```

**注意**: 原始圖片資料集需另外下載 (見下方說明)

---

## 📥 Dataset 下載位置

### 方法 1: 從 Kaggle 下載 (推薦)

1. 前往 [Stanford Cars Dataset on Kaggle](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset)
2. 下載並解壓縮
3. 或使用 Kaggle API：
   ```bash
   kaggle datasets download -d eduardo4jesus/stanford-cars-dataset --unzip
   ```

### 方法 2: 在 Colab 中自動下載

執行 `create_car_indexing.ipynb` 中的 Cell，會自動下載到 `/content/data/`

### 資料集結構

```
data/
├── cars_train/
│   └── cars_train/           # 8,144 張訓練圖片
├── cars_test/
│   └── cars_test/            # 8,041 張測試圖片
└── car_devkit/
    └── devkit/
        ├── cars_meta.mat         # 196 種車款類別名稱
        ├── cars_train_annos.mat  # 訓練集標註
        └── cars_test_annos.mat   # 測試集標註
```

---

## 🔧 建立 Index (使用 Colab)

### Step 1: 開啟 Notebook

在 Google Colab 中開啟 `create_car_indexing.ipynb`

### Step 2: 設定 Kaggle API

```python
# 上傳 kaggle.json
from google.colab import files
uploaded = files.upload()

# 設定認證
import os
os.makedirs('/root/.kaggle', exist_ok=True)
os.rename('kaggle.json', '/root/.kaggle/kaggle.json')
os.chmod('/root/.kaggle/kaggle.json', 0o600)
```

### Step 3: 下載資料集

```python
!kaggle datasets download -d eduardo4jesus/stanford-cars-dataset -p /content/data --unzip
```

### Step 4: 載入 CLIP 模型

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

### Step 5: 產生 Embedding

Notebook 會：
1. 讀取 196 種車款類別名稱
2. 為每張圖片產生 **Image Embedding**
3. 根據車款名稱產生 **Text Embedding**
4. 結合兩者：`Combined = α × Image + (1-α) × Text`

### Step 6: 建立並儲存 FAISS Index

```python
import faiss

# 建立 Index
index_combined = faiss.IndexFlatIP(combined_array.shape[1])
index_combined.add(combined_array)

# 儲存
faiss.write_index(index_combined, "cars_combined.index")
```

### Step 7: 下載 Index 檔案

```python
from google.colab import files

files.download('cars_combined.index')
files.download('cars_image.index')
files.download('cars_text.index')
files.download('cars_metadata.pkl')
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
python search_car.py --index-dir ./index --query "red sports car" --k 5

# 圖片搜尋
python search_car.py --index-dir ./index --image "/path/to/car.jpg" --k 5

# 指定 path.json
python search_car.py --index-dir ./index --path-json ./index/path.json --query "BMW"

# 互動模式
python search_car.py --index-dir ./index --interactive
```

### 方法 2: 在 Python/Notebook 中使用

```python
from search_car import CarSearchEngine

# 初始化搜尋引擎 (模型只載入一次)
engine = CarSearchEngine(
    index_dir="./index",
    path_json="./index/path.json"  # 可選
)

# 文字搜尋
results = engine.search_by_text("red BMW sports car", k=5)
engine.print_results(results)

# 圖片搜尋
results = engine.search_by_image("/path/to/query.jpg", k=5)
engine.print_results(results)

# 在 Notebook 中顯示圖片
engine.show_results_with_images(results)
```

### 方法 3: 互動模式

```bash
python search_car.py --index-dir ./index --interactive
```

```
🎮 進入互動式搜尋模式
   輸入 'q' 或 'quit' 退出
   輸入 't:查詢文字' 進行文字搜尋
   輸入 'i:圖片路徑' 進行圖片搜尋
   輸入 'classes' 列出所有類別

🔍 輸入查詢: BMW SUV
📝 文字搜尋: 'BMW SUV'

============================================================
🔍 搜尋結果:
============================================================

[1] BMW X5 SUV 2007
    Score: 0.3521
    Class ID: 23

[2] BMW X3 SUV 2012
    Score: 0.3498
    Class ID: 22
...
```

---

## 📊 Index 類型說明

| Index 類型 | 說明 | 適用場景 |
|-----------|------|---------|
| `combined` | Image + Text 結合 (預設) | 一般搜尋，最佳平衡 |
| `image` | 純圖片 embedding | 視覺相似度搜尋 |
| `text` | 純文字 embedding | 語義搜尋 |

```python
# 使用不同的 index
results = engine.search_by_text("convertible", k=5, index_type="text")
results = engine.search_by_text("convertible", k=5, index_type="image")
results = engine.search_by_text("convertible", k=5, index_type="combined")
```

---

## ⚙️ 參數設定

### Embedding 權重 (Alpha)

在建立 Index 時設定：
```python
ALPHA = 0.7  # Image 權重 70%, Text 權重 30%
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
| `--index-type` | Index 類型 | `combined` |

---

## 🚀 效能優化

### 模型快取

`search_car.py` 使用全域模型快取，避免重複載入：

```python
# 第一次建立 - 載入模型 (約 60 秒)
engine1 = CarSearchEngine(index_dir="./index")

# 第二次建立 - 使用快取 (瞬間完成)
engine2 = CarSearchEngine(index_dir="./index")
```

### 預先載入模型

```python
from search_car import get_clip_model

# 程式啟動時預先載入
get_clip_model()

# 之後使用都很快
engine = CarSearchEngine(index_dir="./index")
```

---

## 📝 搜尋結果格式

```python
results = engine.search_by_text("BMW", k=3)

# results 是 list of dict
[
    {
        "index": 1234,
        "path": "/path/to/image.jpg",
        "class_id": 23,
        "class_name": "BMW X5 SUV 2007",
        "score": 0.3521
    },
    ...
]
```

---

## 🔗 相關連結

- [Stanford Cars Dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [Kaggle Dataset](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [FAISS](https://github.com/facebookresearch/faiss)

---

## 📄 License

This project uses the Stanford Cars Dataset for academic purposes.

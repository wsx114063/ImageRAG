"""
Stanford Cars Image Retrieval - 搜尋工具
==========================================
使用已儲存的 FAISS Index 進行汽車圖片檢索

使用方式:
    python car_search.py --query "red sports car" --k 5
    python car_search.py --image "/path/to/image.jpg" --k 5
    python car_search.py --interactive
"""

import os
import json
import pickle
import argparse
import numpy as np
import torch
import open_clip
import faiss
from PIL import Image


# ============== 全域模型快取 ==============
_cached_model = None
_cached_preprocess = None
_cached_tokenizer = None
_cached_device = None


def get_clip_model(device=None):
    """
    取得 CLIP 模型 (全域快取，只載入一次)
    
    Args:
        device: 'cuda' 或 'cpu'
    
    Returns:
        (model, preprocess, tokenizer, device)
    """
    global _cached_model, _cached_preprocess, _cached_tokenizer, _cached_device
    
    if _cached_model is not None:
        print("   ✅ 使用已快取的 CLIP 模型")
        return _cached_model, _cached_preprocess, _cached_tokenizer, _cached_device
    
    _cached_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n📦 載入 CLIP 模型 (首次載入，之後會使用快取)...")
    _cached_model, _, _cached_preprocess = open_clip.create_model_and_transforms(
        'ViT-bigG-14',
        pretrained='laion2b_s39b_b160k'
    )
    _cached_model = _cached_model.to(_cached_device)
    _cached_model.eval()
    _cached_tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
    print("   ✅ CLIP 模型載入完成")
    
    return _cached_model, _cached_preprocess, _cached_tokenizer, _cached_device


class CarSearchEngine:
    def __init__(self, index_dir, path_json=None, device=None):
        """
        初始化搜尋引擎
        
        Args:
            index_dir: 儲存 index 和 metadata 的資料夾路徑
            path_json: path.json 檔案路徑 (可選，用於路徑映射)
            device: 'cuda' 或 'cpu'，None 則自動偵測
        """
        # 使用全域快取的模型
        self.model, self.preprocess, self.tokenizer, self.device = get_clip_model(device)
        self.index_dir = index_dir
        self.path_mapping = None
        
        print(f"\n🚗 Stanford Cars Search Engine")
        print(f"   裝置: {self.device}")
        print(f"   Index 路徑: {index_dir}")
        
        # 載入 path.json (如果有)
        if path_json and os.path.exists(path_json):
            self._load_path_json(path_json)
        else:
            # 嘗試在 index_dir 或上層目錄找 path.json
            possible_paths = [
                os.path.join(index_dir, "path.json"),
                os.path.join(os.path.dirname(index_dir), "path.json"),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    self._load_path_json(p)
                    break
        
        # 載入 FAISS Index
        print("\n📦 載入 FAISS Index...")
        self.index_combined = faiss.read_index(os.path.join(index_dir, "cars_combined.index"))
        self.index_image = faiss.read_index(os.path.join(index_dir, "cars_image.index"))
        self.index_text = faiss.read_index(os.path.join(index_dir, "cars_text.index"))
        print(f"   ✅ Combined Index: {self.index_combined.ntotal} 向量")
        print(f"   ✅ Image Index: {self.index_image.ntotal} 向量")
        print(f"   ✅ Text Index: {self.index_text.ntotal} 向量")
        
        # 載入 metadata
        print("\n📦 載入 Metadata...")
        with open(os.path.join(index_dir, "cars_metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
            self.paths = metadata["paths"]
            self.class_ids = metadata["class_ids"]
            self.class_names = metadata["class_names"]
            self.classes = metadata["classes"]
            self.alpha = metadata.get("alpha", 0.7)
        print(f"   ✅ 共 {len(self.paths)} 張圖片")
        print(f"   ✅ 共 {len(self.classes)} 個類別")
        print(f"   ✅ Alpha (image weight): {self.alpha}")
        
        print("\n🎉 搜尋引擎初始化完成！\n")
    
    def _load_path_json(self, path_json):
        """載入 path.json 進行路徑映射"""
        print(f"\n📦 載入 path.json: {path_json}")
        with open(path_json, 'r', encoding='utf-8') as f:
            self.path_mapping = json.load(f)
        
        if isinstance(self.path_mapping, list):
            print(f"   ✅ 載入 {len(self.path_mapping)} 個路徑 (list 格式)")
        elif isinstance(self.path_mapping, dict):
            print(f"   ✅ 載入 {len(self.path_mapping)} 個路徑 (dict 格式)")
    
    def get_image_path(self, index):
        """
        根據索引取得圖片路徑
        優先使用 path.json 的映射，否則使用 metadata 中的路徑
        """
        if self.path_mapping is not None:
            if isinstance(self.path_mapping, list):
                if index < len(self.path_mapping):
                    return self.path_mapping[index]
            elif isinstance(self.path_mapping, dict):
                str_key = str(index)
                if str_key in self.path_mapping:
                    return self.path_mapping[str_key]
        
        # Fallback 到原始路徑
        return self.paths[index]
    
    def _get_index(self, index_type):
        """取得對應的 index"""
        if index_type == "combined":
            return self.index_combined
        elif index_type == "image":
            return self.index_image
        else:
            return self.index_text
    
    def search_by_text(self, query, k=5, index_type="combined"):
        """
        用文字搜尋圖片
        
        Args:
            query: 搜尋文字 (例如: "red sports car", "BMW SUV")
            k: 返回結果數量
            index_type: "combined", "image", "text"
            
        Returns:
            list of dict: 搜尋結果
        """
        # 編碼查詢文字
        text = self.tokenizer([query]).to(self.device)
        with torch.no_grad():
            query_emb = self.model.encode_text(text)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
            query_emb = query_emb.cpu().numpy().astype("float32")
        
        # 搜尋
        index = self._get_index(index_type)
        distances, indices = index.search(query_emb, k)
        
        # 整理結果
        results = []
        for j, i in enumerate(indices[0]):
            results.append({
                "index": int(i),
                "path": self.get_image_path(i),
                "class_id": self.class_ids[i],
                "class_name": self.class_names[i],
                "score": float(distances[0][j])
            })
        return results
    
    def search_by_image(self, img_path, k=5, index_type="combined"):
        """
        用圖片搜尋相似圖片
        
        Args:
            img_path: 查詢圖片路徑
            k: 返回結果數量
            index_type: "combined", "image", "text"
            
        Returns:
            list of dict: 搜尋結果
        """
        # 編碼查詢圖片
        image = self.preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_emb = self.model.encode_image(image)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
            query_emb = query_emb.cpu().numpy().astype("float32")
        
        # 搜尋
        index = self._get_index(index_type)
        distances, indices = index.search(query_emb, k)
        
        # 整理結果
        results = []
        for j, i in enumerate(indices[0]):
            results.append({
                "index": int(i),
                "path": self.get_image_path(i),
                "class_id": self.class_ids[i],
                "class_name": self.class_names[i],
                "score": float(distances[0][j])
            })
        return results
    
    def get_class_info(self, class_id):
        """取得類別資訊"""
        return self.classes.get(class_id, "Unknown")
    
    def list_classes(self):
        """列出所有類別"""
        return self.classes
    
    def print_results(self, results, show_path=True):
        """印出搜尋結果"""
        print("\n" + "=" * 60)
        print("🔍 搜尋結果:")
        print("=" * 60)
        
        for i, r in enumerate(results):
            print(f"\n[{i+1}] {r['class_name']}")
            print(f"    Score: {r['score']:.4f}")
            print(f"    Class ID: {r['class_id']}")
            if show_path:
                print(f"    Path: {r['path']}")
        
        print("\n" + "=" * 60)


def interactive_mode(engine):
    """互動式搜尋模式"""
    print("\n🎮 進入互動式搜尋模式")
    print("   輸入 'q' 或 'quit' 退出")
    print("   輸入 't:查詢文字' 進行文字搜尋")
    print("   輸入 'i:圖片路徑' 進行圖片搜尋")
    print("   輸入 'classes' 列出所有類別")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n🔍 輸入查詢: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("👋 再見！")
                break
            
            if user_input.lower() == 'classes':
                classes = engine.list_classes()
                print(f"\n📋 共 {len(classes)} 個類別:")
                for cid, cname in list(classes.items())[:20]:
                    print(f"   {cid}: {cname}")
                print("   ...")
                continue
            
            # 解析輸入
            if user_input.startswith('t:'):
                query = user_input[2:].strip()
                print(f"\n📝 文字搜尋: '{query}'")
                results = engine.search_by_text(query, k=5)
                engine.print_results(results)
            
            elif user_input.startswith('i:'):
                img_path = user_input[2:].strip()
                if os.path.exists(img_path):
                    print(f"\n🖼️ 圖片搜尋: {img_path}")
                    results = engine.search_by_image(img_path, k=5)
                    engine.print_results(results)
                else:
                    print(f"❌ 找不到圖片: {img_path}")
            
            else:
                # 預設為文字搜尋
                print(f"\n📝 文字搜尋: '{user_input}'")
                results = engine.search_by_text(user_input, k=5)
                engine.print_results(results)
                
        except KeyboardInterrupt:
            print("\n👋 再見！")
            break
        except Exception as e:
            print(f"❌ 錯誤: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Stanford Cars Image Retrieval 搜尋工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 文字搜尋
  python car_search.py --query "red sports car" --k 5
  
  # 圖片搜尋
  python car_search.py --image "/path/to/car.jpg" --k 5
  
  # 指定 index 類型 (combined, image, text)
  python car_search.py --query "BMW" --index-type text --k 10
  
  # 互動模式
  python car_search.py --interactive
  
  # 指定 index 目錄
  python car_search.py --index-dir /path/to/index --query "SUV"
        """
    )
    
    parser.add_argument('--index-dir', type=str, default='./index',
                        help='Index 目錄路徑 (預設: ./index)')
    parser.add_argument('--path-json', type=str, default=None,
                        help='path.json 檔案路徑 (用於路徑映射)')
    parser.add_argument('--query', '-q', type=str,
                        help='文字搜尋查詢')
    parser.add_argument('--image', '-i', type=str,
                        help='圖片搜尋路徑')
    parser.add_argument('--k', type=int, default=5,
                        help='返回結果數量 (預設: 5)')
    parser.add_argument('--index-type', type=str, default='combined',
                        choices=['combined', 'image', 'text'],
                        help='Index 類型 (預設: combined)')
    parser.add_argument('--interactive', action='store_true',
                        help='進入互動模式')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='指定裝置 (預設: 自動偵測)')
    
    args = parser.parse_args()
    
    # 檢查 index 目錄
    if not os.path.exists(args.index_dir):
        print(f"❌ 找不到 index 目錄: {args.index_dir}")
        print("   請確認路徑正確，或使用 --index-dir 指定正確路徑")
        return
    
    # 初始化搜尋引擎
    engine = CarSearchEngine(args.index_dir, path_json=args.path_json, device=args.device)
    
    # 執行搜尋
    if args.interactive:
        interactive_mode(engine)
    
    elif args.query:
        print(f"\n🔍 文字搜尋: '{args.query}'")
        print(f"   Index 類型: {args.index_type}")
        print(f"   返回數量: {args.k}")
        
        results = engine.search_by_text(args.query, k=args.k, index_type=args.index_type)
        engine.print_results(results)
    
    elif args.image:
        if not os.path.exists(args.image):
            print(f"❌ 找不到圖片: {args.image}")
            return
        
        print(f"\n🖼️ 圖片搜尋: {args.image}")
        print(f"   Index 類型: {args.index_type}")
        print(f"   返回數量: {args.k}")
        
        results = engine.search_by_image(args.image, k=args.k, index_type=args.index_type)
        engine.print_results(results)
    
    else:
        # 沒有指定查詢，進入互動模式
        interactive_mode(engine)


if __name__ == "__main__":
    main()

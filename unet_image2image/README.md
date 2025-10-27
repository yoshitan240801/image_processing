# このフォルダのプログラムについて

このフォルダのmainプログラム(main.ipynb)は、U-Netの勉強を兼ねて、Hugging Faceにアップロードされているセマンティックセグメンテーション用のデータをimage-to-imageの題材の代わりとして用いて、U-Netの実装やFine Tuningを試してみたものになります。<br>


# U-Netによるセマンティックセグメンテーション

Pascal VOC 2012データセットを用いた画像セグメンテーションの実装

## プログラム概要

- **目的**: U-Netモデルを用いた画像セグメンテーション
- **データセット**: Pascal VOC 2012 (validation set)
- **フレームワーク**: PyTorch
- **モデル**: カスタムU-Net実装
- **損失関数**: MSE Loss
- **最適化**: Adam optimizer (lr=0.001)

---

## データ準備フロー

```mermaid
flowchart TD
    A[Pascal VOC 2012データセット<br/>ダウンロード] --> B[画像ファイル読み込み]
    B --> C[オリジナル画像<br/>JPEGImages/*.jpg]
    B --> D[セグメント画像<br/>SegmentationClass/*.png]
    C --> E[ファイル名マッチング]
    D --> E
    E --> F[RGB変換 & 3次元チェック]
    F --> G[Train 80% / Valid 20%<br/>データ分割]
```

---

## データ前処理

**データ拡張**
- リサイズ: 224×224ピクセルに統一

**正規化とテンソル化**
1. PIL画像 → NumPy配列
2. 軸の転置: (H, W, C) → (C, H, W)
3. 正規化: ピクセル値を0-1に変換 (÷255)
4. PyTorchテンソルに変換

**カスタムDatasetクラス**
- 入力画像とセグメント画像のペアを管理
- DataLoaderでバッチサイズ20で読み込み

---

## U-Netアーキテクチャ

```mermaid
graph TB
    subgraph Encoder
    A[入力 3×224×224] --> B[Conv Layer 1<br/>64ch]
    B --> C[MaxPool ÷2]
    C --> D[Conv Layer 2<br/>128ch]
    D --> E[MaxPool ÷2]
    E --> F[Conv Layer 3<br/>256ch]
    F --> G[MaxPool ÷2]
    G --> H[Conv Layer 4<br/>512ch]
    H --> I[MaxPool ÷2]
    end
    
    subgraph Bridge
    I --> J[Conv Layer 5<br/>1024ch]
    end
    
    subgraph Decoder
    J --> K[UpConv ×2<br/>512ch]
    K --> L[Conv Layer 6<br/>512ch]
    L --> M[UpConv ×2<br/>256ch]
    M --> N[Conv Layer 7<br/>256ch]
    N --> O[UpConv ×2<br/>128ch]
    O --> P[Conv Layer 8<br/>128ch]
    P --> Q[UpConv ×2<br/>64ch]
    Q --> R[Conv Layer 9<br/>64ch]
    R --> S[出力層<br/>3×224×224]
    end
    
    H -.Skip Connection.-> K
    F -.Skip Connection.-> M
    D -.Skip Connection.-> O
    B -.Skip Connection.-> Q
```

---

## U-Net構成要素

**ConvBlock (畳み込みブロック)**
- Conv2d (3×3, padding=1)
- BatchNorm2d
- ReLU

**ConvLayer (畳み込み層)**
- 3つのConvBlockを連続適用
- チャンネル数を調整しながら特徴抽出

**UpConvBlock (アップサンプリングブロック)**
- ConvTranspose2d (解像度2倍、チャンネル半減)
- BatchNorm2d
- ReLU

---

## データフロー全体像

```mermaid
flowchart TD
    A[Pascal VOC 2012] --> B[データ読み込み]
    B --> C[前処理<br/>リサイズ・正規化]
    C --> D[Dataset/DataLoader]
    D --> E[U-Netモデル]
    E --> F[損失計算<br/>MSE Loss]
    F --> G{学習中?}
    G -->|Yes| H[逆伝播・更新]
    G -->|No| I[予測結果保存]
    H --> J[モデル保存]
    I --> K[逆正規化]
    K --> L[画像表示]
```

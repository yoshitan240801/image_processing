# このフォルダのプログラムについて

このフォルダのmainプログラム(main.ipynb)は、U-Netの勉強を兼ねて、Hugging Faceにアップロードされているセマンティックセグメンテーション用のデータを題材にして、U-Netの実装やFine Tuningを試してみたものになります。<br>


# U-Netによるセマンティックセグメンテーション

Pascal VOC 2012データセットを用いた画像セグメンテーションの実装

## プログラム概要

- 目的: U-Netモデルを使用した画像のセマンティックセグメンテーション
- データセット: Pascal VOC 2012 (validation set)
- フレームワーク: PyTorch
- モデル: カスタムU-Net実装
- クラス数: 22クラス (21クラス + 背景)

---

## データ準備フロー

```mermaid
graph LR
    A[Pascal VOC 2012<br/>ダウンロード] --> B[画像ファイル取得]
    B --> C[オリジナル画像<br/>JPEGImages]
    B --> D[セグメント画像<br/>SegmentationClass]
    C --> E[ファイル名マッチング]
    D --> E
    E --> F[PIL画像リスト作成]
    F --> G[Train/Valid分割<br/>80%/20%]
```

---

## データ前処理

**画像変換処理**
- リサイズ: 224×224ピクセルに統一
- オリジナル画像: 
  - RGB値を255で除算して正規化 (0-1範囲)
  - FloatTensorに変換
- セグメント画像:
  - 値255を21に変換 (背景クラス)
  - LongTensorに変換 (クラスラベル)

---

## U-Netアーキテクチャ (1/3)

**基本ブロック構成**

MyUNetConvBlock: 畳み込みブロック
- Conv2d (3×3, padding=1)
- BatchNorm2d
- ReLU

MyUNetUpConvBlock: アップサンプリングブロック
- ConvTranspose2d (3×3, stride=2)
- BatchNorm2d
- ReLU

---

## U-Netアーキテクチャ (2/3)

```mermaid
graph TD
    A[入力: 3ch, 224×224] --> B[Encoder Layer 1: 64ch]
    B --> C[MaxPool ↓]
    C --> D[Encoder Layer 2: 128ch]
    D --> E[MaxPool ↓]
    E --> F[Encoder Layer 3: 256ch]
    F --> G[MaxPool ↓]
    G --> H[Encoder Layer 4: 512ch]
    H --> I[MaxPool ↓]
    I --> J[Bridge Layer 5: 1024ch]
```

---

## U-Netアーキテクチャ (3/3)

```mermaid
graph TD
    A[Bridge: 1024ch] --> B[UpConv ↑]
    B --> C[Skip Connection]
    C --> D[Decoder Layer 6: 512ch]
    D --> E[UpConv ↑]
    E --> F[Skip Connection]
    F --> G[Decoder Layer 7: 256ch]
    G --> H[UpConv ↑]
    H --> I[Skip Connection]
    I --> J[Decoder Layer 8: 128ch]
    J --> K[UpConv ↑]
    K --> L[Skip Connection]
    L --> M[Decoder Layer 9: 64ch]
    M --> N[出力: 22ch, 224×224]
```

---

## モデル詳細

**ネットワーク構造**
- 入力次元: 3チャンネル (RGB)
- 隠れ層次元: 64 (基準値)
- 出力次元: 22チャンネル (クラス数)
- エンコーダ: 5層 (解像度を1/16に縮小)
- デコーダ: 4層 (解像度を元に戻す)
- Skip Connection: エンコーダとデコーダ間で特徴マップを結合

---

## 推論処理フロー

```mermaid
graph TD
    A[学習済みモデル読み込み] --> B[検証データ入力]
    B --> C[順伝播]
    C --> D[出力: 22チャンネル]
    D --> E[Softmax適用<br/>チャンネル方向]
    E --> F[Argmax適用<br/>最大値チャンネル取得]
    F --> G[予測セグメント画像<br/>1チャンネル]
    G --> H[Tensor→NumPy変換]
    H --> I[PIL画像化]
```

---

## まとめ

**実装内容**
1. Pascal VOC 2012データセットの取得と前処理
2. カスタムDataset/DataLoaderの実装
3. U-Netモデルの実装 (Encoder-Decoder + Skip Connection)
4. 学習・検証ループの実装
5. 損失値の可視化
6. 推論と結果の可視化

# このフォルダのプログラムについて

このフォルダのmainプログラム(main.ipynb)は、pix2pixの勉強を兼ねて、Hugging Faceにアップロードされているセマンティックセグメンテーション用のデータをimage-to-imageの題材の代わりとして用いて、pix2pixの実装やFine Tuningを試してみたものになります。<br>


# Pix2Pix GANによる画像セグメンテーション

Pascal VOC 2012データセットを用いた画像変換モデルの実装

---

## プログラム概要

- **目的**: オリジナル画像からセグメンテーション画像を生成
- **手法**: Pix2Pix (条件付きGAN)
- **データセット**: Pascal VOC 2012 (valデータ)
- **フレームワーク**: PyTorch

```mermaid
graph LR
    A[オリジナル画像] --> B[Generator]
    B --> C[生成画像]
    C --> D[Discriminator]
    E[正解画像] --> D
    D --> F[真偽判定]
```

---

## データ準備フロー

1. Pascal VOC 2012データセットのダウンロード
2. オリジナル画像とセグメンテーション画像のペアを作成
3. RGB形式への変換
4. 80%をtrain、20%をvalidationに分割

**前処理**:
- 画像サイズを224×224にリサイズ
- 正規化 (0-255 → 0-1)
- テンソル化

---

## モデルアーキテクチャ: Generator (U-Net)

```mermaid
graph TD
    A[入力 3x224x224] --> B[Encoder Layer 1: 64ch]
    B --> C[MaxPool ↓]
    C --> D[Encoder Layer 2: 128ch]
    D --> E[MaxPool ↓]
    E --> F[Bridge Layer 3: 256ch]
    F --> G[ConvTranspose ↑]
    G --> H[Decoder Layer 4: 128ch]
    D -.Skip Connection.-> H
    H --> I[ConvTranspose ↑]
    I --> J[Decoder Layer 5: 64ch]
    B -.Skip Connection.-> J
    J --> K[出力 3x224x224]
```

**特徴**: エンコーダとデコーダ間のスキップコネクション

---

## Generator詳細構造

**エンコーダ部分**
- Layer 1: Conv2d(3→64) + BatchNorm + ReLU
- Layer 2: Conv2d(64→128) + BatchNorm + ReLU
- Bridge: Conv2d(128→256) + BatchNorm + ReLU

**デコーダ部分**
- Layer 4: ConvTranspose2d(256→128) + スキップ接続 + Conv2d
- Layer 5: ConvTranspose2d(128→64) + スキップ接続 + Conv2d
- 出力: Conv2d(64→3)

各層でMaxPool2dによるダウンサンプリング、ConvTranspose2dによるアップサンプリング

---

## モデルアーキテクチャ: Discriminator

```mermaid
graph TD
    A[入力 3x224x224] --> B[Conv 16ch + BN + ReLU]
    B --> C[MaxPool]
    C --> D[Conv 32ch + BN + ReLU]
    D --> E[MaxPool]
    E --> F[Conv 64ch + BN + ReLU]
    F --> G[MaxPool]
    G --> H[Conv 128ch + BN + ReLU]
    H --> I[MaxPool]
    I --> J[Conv 256ch + BN + ReLU]
    J --> K[Conv 1ch]
    K --> L[出力 1x14x14]
```

**役割**: 入力画像が本物か生成画像かを判定

---

## 損失関数

**Generator損失**
```python
loss_G = loss_BCE(D(G(x)), true_label) + 100 * loss_MAE(G(x), y)
```
- BCE Loss: Discriminatorを騙すための損失
- MAE Loss: 正解画像との差異を最小化 (係数100)

**Discriminator損失**
```python
loss_D = loss_BCE(D(G(x)), false_label) + loss_BCE(D(y), true_label)
```
- 生成画像を偽物と判定
- 正解画像を本物と判定

---

## 学習プロセス

```mermaid
graph TD
    A[ミニバッチ取得] --> B[Generatorで画像生成]
    B --> C[Discriminatorで判定]
    C --> D[Generator損失計算]
    D --> E[Generatorパラメータ更新]
    E --> F[Discriminatorで再判定]
    F --> G[Discriminator損失計算]
    G --> H[Discriminatorパラメータ更新]
    H --> I{全データ処理完了?}
    I -->|No| A
    I -->|Yes| J[検証フェーズ]
```

---

## まとめ

**実装内容**:
- Pix2Pix GANによる画像変換タスク
- U-Net構造のGeneratorとCNN構造のDiscriminator
- Pascal VOC 2012データセットでの学習

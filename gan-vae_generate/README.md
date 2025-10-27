# このフォルダのプログラムについて

このフォルダのmainプログラム(main.ipynb)は、GANやVAEの勉強を兼ねて、Hugging Faceにアップロードされているデータを画像生成の題材として用いて、GANやVAEの実装およびFine Tuningを試してみたものになります。<br>


# VAE-GANによるアニメ顔画像生成

## プログラムの目的

**VAE (Variational Autoencoder) と GAN (Generative Adversarial Network) を組み合わせたモデルによるアニメ顔画像の生成**

- HuggingFaceのアニメ顔データセットを使用
- VAEとGANのハイブリッドアーキテクチャで学習
- 潜在変数から新規画像を生成

---

## データ準備フロー

```mermaid
flowchart TD
    A[HuggingFaceからデータセット読み込み] --> B[3階テンソルのみ抽出]
    B --> C[3チャンネルRGB画像のみ抽出]
    C --> D[データ拡張の定義]
    D --> E[正規化とテンソル化]
    E --> F[DataLoaderの作成]
```

**データ拡張内容:**
- リサイズ: 96×96
- ランダム水平反転 (p=0.5)
- ランダム回転 (-10°～10°)

---

## モデルアーキテクチャ (1/3): Generator

```mermaid
flowchart LR
    A[入力画像<br/>3×96×96] --> B[Encoder]
    B --> C[潜在変数 z<br/>256次元]
    C --> D[Decoder]
    D --> E[出力画像<br/>3×96×96]
    B -.-> F[μ, σ]
    F -.-> C
```

**Encoder:**
- 5層の畳み込み層 (Conv2d + BatchNorm + LeakyReLU)
- チャンネル数: 3→32→64→128→256→512
- 解像度: 96×96 → 3×3
- 全結合層でμとσを出力し、再パラメータ化トリックでzを生成

---

## モデルアーキテクチャ (2/3): Decoder

**Decoder:**
- 全結合層で潜在変数を展開
- 5層の転置畳み込み層 (ConvTranspose2d + BatchNorm + LeakyReLU)
- チャンネル数: 512→256→128→64→32→3
- 解像度: 3×3 → 96×96
- 最終層で3チャンネルの画像を出力

---

## モデルアーキテクチャ (3/3): Discriminator

```mermaid
flowchart LR
    A[入力画像<br/>3×96×96] --> B[Conv層×5]
    B --> C[特徴マップ<br/>512×3×3]
    C --> D[1×1 Conv]
    D --> E[判定結果<br/>1×3×3]
```

**Discriminator:**
- 5層の畳み込み層 (Conv2d + BatchNorm + LeakyReLU)
- チャンネル数: 3→32→64→128→256→512
- 最終層で真偽判定を出力

---

## 損失関数の構成

**Generator (G) の損失:**
- `loss_G1`: GAN損失 (DがG出力を本物と判定するように)
- `loss_G2`: L1損失 (入力画像との差) × 係数100
- `loss_G3`: VAE再構成損失 (BCE)
- `loss_G4`: VAE正則化項 (KLダイバージェンス)

**Discriminator (D) の損失:**
- `loss_D1`: 本物画像を本物と判定
- `loss_D2`: G生成画像を偽物と判定

---

## 学習プロセス

```mermaid
flowchart TD
    A[ミニバッチ取得] --> B[Gで画像生成]
    B --> C[Dで真偽判定]
    C --> D[G損失計算]
    D --> E[G更新]
    E --> F[D損失計算]
    F --> G[D更新]
    G --> H{全データ処理完了?}
    H -->|No| A
    H -->|Yes| I[エポック終了]
    I --> J[モデル保存]
```

---

## 画像生成プロセス

```mermaid
flowchart LR
    A[学習済み潜在変数群] --> B[ランダム重み付け平均]
    B --> C[新規潜在変数 z]
    C --> D[Decoder]
    D --> E[生成画像<br/>3×96×96]
    E --> F[Sigmoid適用]
    F --> G[非正規化]
    G --> H[PIL画像化]
```

**生成手順:**
1. 最新エポックの潜在変数を取得
2. ランダムな重みで100個の潜在変数を生成
3. Decoderで画像生成
4. 10×10グリッドで可視化

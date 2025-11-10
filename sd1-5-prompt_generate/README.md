# このフォルダのプログラムについて

このフォルダのmainプログラム(main.ipynb)は、Stable Diffusion 1.5のFine TunedモデルのSomethingV2_2を用いて、プロンプトにて画像生成を試してみたものになります。<br>


# 画像生成AIプログラムの概要

diffusersライブラリーでstable diffusionを使用した画像生成プログラム

---

## 処理フロー

```mermaid
flowchart TD
    A[開始] --> B[デバイス設定<br/>CUDA or CPU]
    B --> C[モデルロード<br/>NoCrypt/SomethingV2_2]
    C --> D[パイプラインをデバイスに転送]
    D --> E[画像生成実行<br/>プロンプト指定]
    E --> F[10枚の画像を生成]
    F --> G[matplotlibで一覧表示<br/>2行5列]
    G --> H[IPythonで個別表示]
    H --> I[終了]
```

---

## モデル設定

- **デバイス**: CUDA利用可能ならGPU、なければCPU
- **モデル**: `NoCrypt/SomethingV2_2` (Hugging Faceからロード)
- **データ型**: `torch.float16` (メモリ効率化)

---

## 画像生成パラメータ

| パラメータ | 値 | 説明 |
|----------|-----|------|
| `height` / `width` | 512 | 画像サイズ |
| `num_inference_steps` | 20 | 推論ステップ数 |
| `guidance_scale` | 7.0 | プロンプト遵守度 |
| `num_images_per_prompt` | 10 | 生成画像数 |
| `generator` | seed=1234 | 乱数シード(再現性) |
| `output_type` | "pil" | 出力形式 |

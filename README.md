# 🧠 画像処理・ディープラーニング実装集  
**（ResNet・U-Net・Vision Transformer・VAE・GAN・Pix2Pix・Stable Diffusion など）**

このリポジトリは、画像処理や生成モデルの代表的なアーキテクチャをPyTorchで自学実装したり、事前学習済みモデルを利用したりした実験集です。  
分類・セグメンテーション・画像変換・生成などのタスクになります。

---

## 🔧 使用技術

- **PyTorch** — モデル実装・学習フレームワーク  
- **Torchvision** — 画像前処理・データセット管理
- **diffusers** — 拡散モデル  
- **OpenCV** — 画像読み込みや可視化  
- **Matplotlib / NumPy** — 結果の可視化・数値処理  

---

## 📁 ディレクトリ構成

| ディレクトリ名 | 概要 |
|----------------|------|
| [`resnet50_classification`](./resnet50_classification) | ResNet-50 を用いた画像分類タスクの実装 |
| [`unet_segmentation`](./unet_segmentation) | U-Net による画像セグメンテーションの実装 |
| [`vision-transformer_classification`](./vision-transformer_classification) | Vision Transformer (ViT) による画像分類タスク |
| [`unet_image2image`](./unet_image2image) | U-Net を利用した画像変換（image-to-image translation）の実験 |
| [`vae_generate`](./vae_generate) | VAE（Variational Autoencoder）による画像生成 |
| [`gan-vae_generate`](./gan-vae_generate) | GAN と VAE のハイブリッドモデルによる生成実験 |
| [`gan-vae-resnet_generate`](./gan-vae-resnet_generate) | ResNet構造を取り入れたGAN-VAEによる高品質生成 |
| [`pix2pix_image2image`](./pix2pix_image2image) | Pix2Pix を用いた条件付きGANによる画像変換 |
| [`sd1-5-prompt_generate`](./sd1-5-prompt_generate) | diffusersでstable diffusionを用いた画像生成 |
| [`sd1-5-prompt_generate_2`](./sd1-5-prompt_generate_2) | diffusersでstable diffusionのモデルマージによる画像生成 |
| [`sd1-5-prompt_generate_3`](./sd1-5-prompt_generate_3) | diffusersでstable diffusionのプロンプトトークンや強調をカスタムした画像生成 |

---

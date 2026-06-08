<div align="center">

<img src="https://raw.githubusercontent.com/nashory/gans-awesome-applications/master/jpg/gans.jpg" alt="gans-awesome-applications banner" width="100%">

# gans-awesome-applications

**A curated list of awesome GAN _applications_ and demonstrations.**

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
![Last Commit](https://img.shields.io/github/last-commit/nashory/gans-awesome-applications)
![Stars](https://img.shields.io/github/stars/nashory/gans-awesome-applications?style=social)

</div>

> **Only awesome is awesome. This is a curation, not a collection.**
>
> This list is about what GANs are *used for*. General GAN papers whose contribution is the model
> itself (DCGAN, BEGAN, WGAN, …) are **not** included as entries — only genuine landmarks make the
> short list at the top. Pure theory / loss-function papers are out of scope; diffusion-only work is
> out of scope (GAN+diffusion hybrids are fine).

🏷️ **Legend** — each entry links to what exists, in this order:
`[[paper]]` · `[[github]]` code · `[[project]]` project page · `[[demo]]`/`[[colab]]` · `[[video]]`/`[[youtube]]` · `[[blog]]`.

🤝 Contributions welcome — see **[CONTRIBUTING.md](CONTRIBUTING.md)**.

-----

## The landmark papers that I respect.
+ Generative Adversarial Networks (NeurIPS 2014), [[paper]](https://arxiv.org/abs/1406.2661), [[github]](https://github.com/goodfeli/adversarial)
+ Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (ICLR 2016), [[paper]](https://arxiv.org/pdf/1511.06434), [[github]](https://github.com/soumith/dcgan.torch)
+ Improved Techniques for Training GANs (NeurIPS 2016), [[paper]](https://arxiv.org/pdf/1606.03498.pdf), [[github]](https://github.com/openai/improved-gan)
+ BEGAN: Boundary Equilibrium Generative Adversarial Networks (2017), [[paper]](https://arxiv.org/pdf/1703.10717), [[github]](https://github.com/carpedm20/BEGAN-tensorflow)
+ Training Generative Adversarial Networks with Limited Data (StyleGAN2-ADA) (NeurIPS 2020), [[paper]](https://arxiv.org/abs/2006.06676), [[github]](https://github.com/NVlabs/stylegan2-ada-pytorch)
+ The GAN is dead; long live the GAN! A Modern GAN Baseline (R3GAN) (NeurIPS 2024), [[paper]](https://arxiv.org/abs/2501.05441), [[github]](https://github.com/brownvc/R3GAN)

-----

## Contents
<details>
<summary><b>Click to expand the table of contents</b> — or just press <kbd>command</kbd>/<kbd>ctrl</kbd> + <kbd>F</kbd> to search for a keyword.</summary>

+ [Applications using GANs](#applications-using-gans)
    + [Font generation](#font-generation)
    + [Anime character generation](#anime-character-generation)
    + [Face Stylization](#face-stylization)
    + [Interactive Image generation](#interactive-image-generation)
    + [GAN Inversion and Latent Space Editing](#gan-inversion-and-latent-space-editing)
    + [Text2Image (text to image)](#text2image-text-to-image)
    + [Adversarial Diffusion Distillation (GAN-hybrid)](#adversarial-diffusion-distillation-gan-hybrid)
    + [3D Object generation](#3d-object-generation)
    + [3D-aware Image Synthesis](#3d-aware-image-synthesis)
    + [Image Editing](#image-editing)
    + [Face Aging](#face-aging)
    + [Human Pose Estimation](#human-pose-estimation)
    + [Talking Head and Face Reenactment](#talking-head-and-face-reenactment)
    + [Virtual Try-On](#virtual-try-on)
    + [Domain-transfer (e.g. style-transfer, pix2pix, sketch2image)](#domain-transfer-eg-style-transfer-pix2pix-sketch2image)
    + [Image Inpainting (hole filling)](#image-inpainting-hole-filling)
    + [Image Blending](#image-blending)
    + [Super-resolution](#super-resolution)
    + [High-resolution image generation (large-scale image)](#high-resolution-image-generation-large-scale-image)
    + [Adversarial Examples (Defense vs Attack)](#adversarial-examples-defense-vs-attack)
    + [Visual Saliency Prediction (attention prediction)](#visual-saliency-prediction-attention-prediction)
    + [Object Detection/Recognition](#object-detectionrecognition)
    + [Robotics](#robotics)
    + [Video (generation/prediction)](#video-generationprediction)
    + [Audio and Speech Synthesis](#audio-and-speech-synthesis)
    + [Text and Sequence Generation](#text-and-sequence-generation)
    + [Anomaly Detection](#anomaly-detection)
    + [Synthetic Data Generation](#synthetic-data-generation)
    + [Others](#others)
+ [Did not use GAN, but still interesting applications](#did-not-use-gan-but-still-interesting-applications)
    + [Real-time face reconstruction](#real-time-face-reconstruction)
    + [Super-resolution](#super-resolution-1)
    + [Photorealistic Image generation (e.g. pix2pix, sketch2image)](#photorealistic-image-generation-eg-pix2pix-sketch2image)
    + [Human Pose Estimation](#human-pose-estimation-1)
    + [3D Object generation](#3d-object-generation-1)
+ [GAN tutorials with easy and simple example code for starters](#gan-tutorials-with-easy-and-simple-example-code-for-starters)
+ [Implementations of various types of GANs collection](#implementations-of-various-types-of-gans-collection)
+ [Trendy AI-application Articles](#trendy-ai-application-articles)

</details>

-----

## Applications using GANs

### Font generation
+ Learning Chinese Character style with conditional GAN (zi2zi) (2017), [[blog]](https://kaonashi-tyc.github.io/2017/04/06/zi2zi.html), [[github]](https://github.com/kaonashi-tyc/zi2zi)
+ Artistic Glyph Image Synthesis via One-Stage Few-Shot Learning (AGIS-Net) (SIGGRAPH Asia 2019), [[paper]](http://arxiv.org/abs/1910.04987), [[github]](https://github.com/hologerry/AGIS-Net)
+ Attribute2Font: Creating Fonts You Want From Attributes (SIGGRAPH 2020), [[paper]](https://arxiv.org/abs/2005.07865), [[github]](https://github.com/hologerry/Attr2Font)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Anime character generation
+ Towards the Automatic Anime Characters Creation with Generative Adversarial Networks (2017), [[paper]](https://arxiv.org/pdf/1708.05509)
+ [Project] A simple PyTorch Implementation of Generative Adversarial Networks, focusing on anime face drawing, [[github]](https://github.com/jayleicn/animeGAN)
+ [Project] A simple, clean TensorFlow implementation of Generative Adversarial Networks with a focus on modeling illustrations, [[github]](https://github.com/tdrussell/IllustrationGAN)
+ [Project] Keras-GAN-Animeface-Character, [[github]](https://github.com/forcecore/Keras-GAN-Animeface-Character)
+ [Project] A DCGAN to generate anime faces using custom mined dataset, [[github]](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Face Stylization
+ JoJoGAN: One Shot Face Stylization (ECCV 2022), [[paper]](https://arxiv.org/abs/2112.11641), [[github]](https://github.com/mchong6/JoJoGAN)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Interactive Image generation
+ Generative Visual Manipulation on the Natural Image Manifold (iGAN) (ECCV 2016), [[paper]](https://arxiv.org/pdf/1609.03552), [[github]](https://github.com/junyanz/iGAN)
+ Neural Photo Editing with Introspective Adversarial Networks (ICLR 2017), [[paper]](http://arxiv.org/abs/1609.07093), [[github]](https://github.com/ajbrock/Neural-Photo-Editor)
+ Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold (DragGAN) (SIGGRAPH 2023), [[paper]](https://arxiv.org/abs/2305.10973), [[github]](https://github.com/XingangPan/DragGAN)

<sub><a href="#contents">↑ back to Contents</a></sub>

### GAN Inversion and Latent Space Editing
+ StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery (ICCV 2021), [[paper]](https://arxiv.org/abs/2103.17249), [[github]](https://github.com/orpatashnik/StyleCLIP)
+ Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation (pSp) (CVPR 2021), [[paper]](https://arxiv.org/abs/2008.00951), [[github]](https://github.com/eladrich/pixel2style2pixel)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Text2Image (text to image)
+ TAC-GAN – Text Conditioned Auxiliary Classifier Generative Adversarial Network (2017), [[paper]](https://arxiv.org/pdf/1703.06412.pdf), [[github]](https://github.com/dashayushman/TAC-GAN)
+ StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks (ICCV 2017), [[paper]](https://arxiv.org/pdf/1612.03242.pdf), [[github]](https://github.com/hanzhanggit/StackGAN)
+ Generative Adversarial Text to Image Synthesis (ICML 2016), [[paper]](https://arxiv.org/pdf/1605.05396.pdf), [[github]](https://github.com/paarthneekhara/text-to-image), [[github]](https://github.com/reedscot/icml2016)
+ Learning What and Where to Draw (NeurIPS 2016), [[paper]](http://www.scottreed.info/files/nips2016.pdf), [[github]](https://github.com/reedscot/nips2016)
+ AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks (CVPR 2018), [[paper]](https://arxiv.org/abs/1711.10485), [[github]](https://github.com/taoxugit/AttnGAN)
+ GigaGAN: Scaling up GANs for Text-to-Image Synthesis (CVPR 2023), [[paper]](https://arxiv.org/abs/2303.05511)
+ StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis (ICML 2023), [[paper]](https://arxiv.org/abs/2301.09515), [[github]](https://github.com/autonomousvision/stylegan-t)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Adversarial Diffusion Distillation (GAN-hybrid)
_GANs strike back in 2024–2026: adversarial objectives distill slow diffusion samplers into one/few-step generators._
+ UFOGen: You Forward Once Large Scale Text-to-Image Generation via Diffusion GANs (CVPR 2024), [[paper]](https://arxiv.org/abs/2311.09257)
+ Improved Distribution Matching Distillation for Fast Image Synthesis (DMD2) (NeurIPS 2024), [[paper]](https://arxiv.org/abs/2405.14867), [[github]](https://github.com/tianweiy/DMD2)

<sub><a href="#contents">↑ back to Contents</a></sub>

### 3D Object generation
+ Parametric 3D Exploration with Stacked Adversarial Networks (2016), [[github]](https://github.com/maxorange/pix2vox), [[youtube]](https://www.youtube.com/watch?v=ITATOXVvWEM)
+ Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (3D-GAN) (NeurIPS 2016), [[paper]](http://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf), [[github]](https://github.com/zck119/3dgan-release), [[youtube]](https://www.youtube.com/watch?v=HO1LYJb818Q)
+ 3D Shape Induction from 2D Views of Multiple Objects (2016), [[paper]](https://arxiv.org/pdf/1612.05872.pdf)
+ Fully Convolutional Refined Auto-Encoding Generative Adversarial Networks for 3D Multi Object Scenes (2017), [[github]](https://github.com/yunishi3/3D-FCR-alphaGAN), [[blog]](https://becominghuman.ai/3d-multi-object-gan-7b7cee4abf80)

<sub><a href="#contents">↑ back to Contents</a></sub>

### 3D-aware Image Synthesis
+ Efficient Geometry-aware 3D Generative Adversarial Networks (EG3D) (CVPR 2022), [[paper]](https://arxiv.org/abs/2112.07945), [[github]](https://github.com/NVlabs/eg3d)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Image Editing
+ Invertible Conditional GANs for image editing (IcGAN) (2016), [[paper]](https://arxiv.org/abs/1611.06355), [[github]](https://github.com/Guim3/IcGAN)
+ Image De-raining Using a Conditional Generative Adversarial Network (ID-CGAN) (2017), [[paper]](https://arxiv.org/abs/1701.05957), [[github]](https://github.com/hezhangsprinter/ID-CGAN)
+ DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks (CVPR 2018), [[paper]](https://arxiv.org/abs/1711.07064), [[github]](https://github.com/KupynOrest/DeblurGAN)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Face Aging
+ Age Progression/Regression by Conditional Adversarial Autoencoder (CAAE) (CVPR 2017), [[paper]](https://arxiv.org/pdf/1702.08423), [[github]](https://github.com/ZZUTK/Face-Aging-CAAE)
+ CAN: Creative Adversarial Networks Generating “Art” by Learning About Styles and Deviating from Style Norms (2017), [[paper]](https://arxiv.org/pdf/1706.07068.pdf)
+ FACE AGING WITH CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS (2017), [[paper]](https://arxiv.org/pdf/1702.01983.pdf)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Human Pose Estimation
+ Joint Discriminative and Generative Learning for Person Re-identification (DG-Net) (CVPR 2019), [[paper]](https://arxiv.org/abs/1904.07223), [[github]](https://github.com/NVlabs/DG-Net), [[video]](https://www.youtube.com/watch?v=ubCrEAIpQs4)
+ Pose Guided Person Image Generation (NeurIPS 2017), [[paper]](https://arxiv.org/abs/1705.09368)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Talking Head and Face Reenactment
+ First Order Motion Model for Image Animation (NeurIPS 2019), [[paper]](https://arxiv.org/abs/2003.00196), [[github]](https://github.com/AliaksandrSiarohin/first-order-model)
+ A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild (Wav2Lip) (ACM MM 2020), [[paper]](https://arxiv.org/abs/2008.10010), [[github]](https://github.com/Rudrabha/Wav2Lip)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Virtual Try-On
+ VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization (CVPR 2021), [[paper]](https://arxiv.org/abs/2103.16874), [[github]](https://github.com/shadow2496/VITON-HD)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Domain-transfer (e.g. style-transfer, pix2pix, sketch2image)
+ Image-to-Image Translation with Conditional Adversarial Networks (pix2pix) (CVPR 2017), [[paper]](https://arxiv.org/pdf/1611.07004), [[github]](https://github.com/phillipi/pix2pix), [[youtube]](https://www.youtube.com/watch?v=VVqxbmUJorQ)
+ Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN) (ICCV 2017), [[paper]](https://arxiv.org/pdf/1703.10593.pdf), [[github]](https://github.com/junyanz/CycleGAN), [[youtube]](https://www.youtube.com/watch?v=JzgOfISLNjk)
+ Learning to Discover Cross-Domain Relations with Generative Adversarial Networks (DiscoGAN) (ICML 2017), [[paper]](https://arxiv.org/pdf/1703.05192.pdf), [[github]](https://github.com/carpedm20/DiscoGAN-pytorch)
+ StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation (CVPR 2018), [[paper]](https://arxiv.org/abs/1711.09020), [[github]](https://github.com/yunjey/stargan)
+ StarGAN v2: Diverse Image Synthesis for Multiple Domains (CVPR 2020), [[paper]](https://arxiv.org/abs/1912.01865), [[github]](https://github.com/clovaai/stargan-v2)
+ Multimodal Unsupervised Image-to-Image Translation (MUNIT) (ECCV 2018), [[paper]](https://arxiv.org/abs/1804.04732), [[github]](https://github.com/NVlabs/MUNIT)
+ Unsupervised Creation of Parameterized Avatars (2017), [[paper]](https://arxiv.org/pdf/1704.05693.pdf)
+ Unsupervised Cross-Domain Image Generation (DTN) (ICLR 2017), [[paper]](https://openreview.net/pdf?id=Sk2Im59ex)
+ Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks (MGANs) (ECCV 2016), [[paper]](http://arxiv.org/abs/1604.04382), [[github]](https://github.com/chuanli11/MGANs)
+ Pixel-Level Domain Transfer (PixelDTGAN) (ECCV 2016), [[paper]](https://arxiv.org/pdf/1603.07442), [[github]](https://github.com/fxia22/PixelDTGAN)
+ TextureGAN: Controlling Deep Image Synthesis with Texture Patches (CVPR 2018), [[paper]](https://arxiv.org/pdf/1706.02823.pdf), [[demo]](https://github.com/varunagrawal/t-gan-demo)
+ Vincent AI Sketch Demo Draws In Throngs at GTC Europe (2017), [[blog]](https://blogs.nvidia.com/blog/2017/10/11/vincent-ai-sketch-demo-draws-in-throngs-at-gtc-europe/), [[youtube]](https://www.youtube.com/watch?v=kIcqXTUMwps)
+ Deep Photo Style Transfer (CVPR 2017), [[paper]](https://arxiv.org/pdf/1703.07511.pdf), [[github]](https://github.com/luanfujun/deep-photo-styletransfer)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Image Inpainting (hole filling)
+ Context Encoders: Feature Learning by Inpainting (CVPR 2016), [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Pathak_Context_Encoders_Feature_CVPR_2016_paper.pdf), [[github]](https://github.com/pathak22/context-encoder)
+ Semantic Image Inpainting with Perceptual and Contextual Losses (CVPR 2017), [[paper]](https://arxiv.org/abs/1607.07539), [[github]](https://github.com/bamos/dcgan-completion.tensorflow)
+ SEMI-SUPERVISED LEARNING WITH CONTEXT-CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS (2016), [[paper]](https://arxiv.org/pdf/1611.06430v1.pdf)
+ Generative Face Completion (CVPR 2017), [[paper]](https://drive.google.com/file/d/0B8_MZ8a8aoSeenVrYkpCdnFRVms/edit), [[github]](https://github.com/Yijunmaverick/GenerativeFaceCompletion)
+ Free-Form Image Inpainting with Gated Convolution (DeepFill v2) (ICCV 2019), [[paper]](https://arxiv.org/abs/1806.03589), [[github]](https://github.com/JiahuiYu/generative_inpainting)
+ Resolution-robust Large Mask Inpainting with Fourier Convolutions (LaMa) (WACV 2022), [[paper]](https://arxiv.org/abs/2109.07161), [[github]](https://github.com/advimman/lama)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Image Blending
+ GP-GAN: Towards Realistic High-Resolution Image Blending (ACM MM 2019), [[paper]](https://arxiv.org/abs/1703.07195), [[github]](https://github.com/wuhuikai/GP-GAN)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Super-resolution
+ Image super-resolution through deep learning (srez) (2016), [[github]](https://github.com/david-gpu/srez)
+ Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN) (CVPR 2017), [[paper]](https://arxiv.org/abs/1609.04802), [[github]](https://github.com/leehomyc/Photo-Realistic-Super-Resoluton)
+ High-Quality Face Image Super-Resolution Using Conditional Generative Adversarial Networks (2017), [[paper]](https://arxiv.org/pdf/1707.00737.pdf)
+ Analyzing Perception-Distortion Tradeoff using Enhanced Perceptual Super-resolution Network (EPSR) (ECCVW 2018), [[paper]](https://arxiv.org/pdf/1811.00344.pdf), [[github]](https://github.com/subeeshvasu/2018_subeesh_epsr_eccvw)
+ ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (ECCVW 2018), [[paper]](https://arxiv.org/abs/1809.00219), [[github]](https://github.com/xinntao/ESRGAN)
+ Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data (ICCVW 2021), [[paper]](https://arxiv.org/abs/2107.10833), [[github]](https://github.com/xinntao/Real-ESRGAN)

<sub><a href="#contents">↑ back to Contents</a></sub>

### High-resolution image generation (large-scale image)
+ Generating Large Images from Latent Vectors (2016), [[blog]](http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/), [[github]](https://github.com/hardmaru/cppn-gan-vae-tensorflow)
+ Progressive Growing of GANs for Improved Quality, Stability, and Variation (PGGAN) (ICLR 2018), [[paper]](http://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of//karras2017gan-paper.pdf), [[github]](https://github.com/tkarras/progressive_growing_of_gans)
+ Large Scale GAN Training for High Fidelity Natural Image Synthesis (BigGAN) (ICLR 2019), [[paper]](https://arxiv.org/abs/1809.11096)
+ SinGAN: Learning a Generative Model from a Single Natural Image (ICCV 2019), [[paper]](https://arxiv.org/abs/1905.01164), [[github]](https://github.com/tamarott/SinGAN)
+ Analyzing and Improving the Image Quality of StyleGAN (StyleGAN2) (CVPR 2020), [[paper]](https://arxiv.org/abs/1912.04958), [[github]](https://github.com/NVlabs/stylegan2)
+ Alias-Free Generative Adversarial Networks (StyleGAN3) (NeurIPS 2021), [[paper]](https://arxiv.org/abs/2106.12423), [[github]](https://github.com/NVlabs/stylegan3)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Adversarial Examples (Defense vs Attack)
+ SafetyNet: Detecting and Rejecting Adversarial Examples Robustly (ICCV 2017), [[paper]](https://arxiv.org/abs/1704.00103)
+ ADVERSARIAL EXAMPLES FOR GENERATIVE MODELS (2017), [[paper]](https://arxiv.org/pdf/1702.06832.pdf)
+ Adversarial Examples Generation and Defense Based on Generative Adversarial Network (2016), [[paper]](http://cs229.stanford.edu/proj2016/report/LiuXia-AdversarialExamplesGenerationAndDefenseBasedOnGenerativeAdversarialNetwork-report.pdf)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Visual Saliency Prediction (attention prediction)
+ SalGAN: Visual Saliency Prediction with Generative Adversarial Networks (2017), [[paper]](https://arxiv.org/pdf/1701.01081), [[github]](https://github.com/imatge-upc/saliency-salgan-2017)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Object Detection/Recognition
+ Perceptual Generative Adversarial Networks for Small Object Detection (CVPR 2017), [[paper]](https://arxiv.org/pdf/1706.05274)
+ Adversarial Generation of Training Examples for Vehicle License Plate Recognition (2017), [[paper]](https://arxiv.org/pdf/1707.03124.pdf)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Robotics
+ Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks (PixelDA) (CVPR 2017), [[paper]](https://arxiv.org/pdf/1612.05424.pdf), [[github]](https://github.com/rhythm92/Unsupervised-Pixel-Level-Domain-Adaptation-with-GAN)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Video (generation/prediction)
+ Deep Multi-Scale Video Prediction Beyond Mean Square Error (ICLR 2016), [[paper]](https://arxiv.org/pdf/1511.05440.pdf), [[github]](https://github.com/dyelax/Adversarial_Video_Generation)
+ Learning Temporal Coherence via Self-Supervision for GAN-based Video Generation (TecoGAN) (SIGGRAPH 2020), [[paper]](https://arxiv.org/abs/1811.09393), [[github]](https://github.com/thunil/TecoGAN)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Audio and Speech Synthesis
+ Adversarial Audio Synthesis (WaveGAN) (ICLR 2019), [[paper]](https://arxiv.org/abs/1802.04208), [[github]](https://github.com/chrisdonahue/wavegan)
+ HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis (NeurIPS 2020), [[paper]](https://arxiv.org/abs/2010.05646), [[github]](https://github.com/jik876/hifi-gan)
+ BigVGAN: A Universal Neural Vocoder with Large-Scale Training (ICLR 2023, v2 2024), [[paper]](https://arxiv.org/abs/2206.04658), [[github]](https://github.com/NVIDIA/BigVGAN)
+ Vocos: Closing the Gap between Time-domain and Fourier-based Neural Vocoders (ICLR 2024), [[paper]](https://arxiv.org/abs/2306.00814), [[github]](https://github.com/gemelo-ai/vocos)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Text and Sequence Generation
+ SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient (AAAI 2017), [[paper]](https://arxiv.org/abs/1609.05473), [[github]](https://github.com/LantaoYu/SeqGAN)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Anomaly Detection
+ Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (AnoGAN) (IPMI 2017), [[paper]](https://arxiv.org/abs/1703.05921)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Synthetic Data Generation
+ Learning from Simulated and Unsupervised Images through Adversarial Training (SimGAN) (CVPR 2017), [[paper]](https://arxiv.org/pdf/1612.07828.pdf), [[github]](https://github.com/carpedm20/simulated-unsupervised-tensorflow)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Others
+ (Physics) Learning Particle Physics by Example: Location-Aware Generative Adversarial Networks for Physics Synthesis (2017), [[paper]](https://arxiv.org/pdf/1701.05927.pdf), [[github]](https://github.com/hep-lbdl/adversarial-jets)
+ (Games) Style Transfer Generative Adversarial Networks: Learning to Play Chess Differently (2017), [[paper]](https://openreview.net/pdf?id=HkpbnufYe), [[github]](https://github.com/2014mchidamb/AdversarialChess)
+ (General) Spectral Normalization for Generative Adversarial Networks (ICLR 2018), [[paper]](https://openreview.net/pdf?id=B1QRgziT-), [[github]](https://github.com/minhnhat93/tf-SNDCGAN)

<sub><a href="#contents">↑ back to Contents</a></sub>

-----

## Did not use GAN, but still interesting applications.

### Real-time face reconstruction
+ Model-based Deep Convolutional Face Autoencoder for Unsupervised Monocular Reconstruction (MoFA) (ICCV 2017), [[paper]](https://arxiv.org/pdf/1703.10580.pdf), [[github]](https://github.com/waxz/MoFA), [[youtube]](https://www.youtube.com/watch?v=uIMpHZYB8fI)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Super-resolution
+ Learning to Simplify: Fully Convolutional Networks for Rough Sketch Cleanup (SIGGRAPH 2016), [[site link]](http://hi.cs.waseda.ac.jp/~esimo/en/research/sketch/), [[youtube]](https://www.youtube.com/watch?v=4MfG9CDufPA)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Photorealistic Image generation (e.g. pix2pix, sketch2image)
+ The Sketchy Database: Learning to Retrieve Badly Drawn Bunnies (SIGGRAPH 2016), [[youtube]](https://www.youtube.com/watch?v=a3sgFQjEfp4)
+ PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing (SIGGRAPH 2009), [[paper]](https://www.researchgate.net/profile/Eli_Shechtman/publication/220184392_PatchMatch_A_Randomized_Correspondence_Algorithm_for_Structural_Image_Editing/links/02e7e520897b12bf0f000000.pdf), [[github]](https://github.com/younesse-cv/PatchMatch), [[youtube]](https://www.youtube.com/watch?v=n3aoc36V8LM)

<sub><a href="#contents">↑ back to Contents</a></sub>

### Human Pose Estimation
+ Knowledge-Guided Deep Fractal Neural Networks for Human Pose Estimation (2017), [[paper]](https://arxiv.org/pdf/1705.02407.pdf), [[github]](https://github.com/Guanghan/GNet-pose)

<sub><a href="#contents">↑ back to Contents</a></sub>

### 3D Object generation
+ 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction (ECCV 2016), [[paper]](http://arxiv.org/abs/1604.00449), [[github]](https://github.com/chrischoy/3D-R2N2)

<sub><a href="#contents">↑ back to Contents</a></sub>

-----

## GAN tutorials with easy and simple example code for starters
+ [1D Generative Adversarial Network Demo](http://notebooks.aylien.com/research/gan/gan_simple.html)
+ [starter from "How to Train a GAN?" at NIPS2016](https://github.com/soumith/ganhacks)
+ [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)
+ [OpenAI - Generative Models](https://blog.openai.com/generative-models/)

<sub><a href="#contents">↑ back to Contents</a></sub>

----

## Implementations of various types of GANs collection
+ [nashory/gans-collections.torch](https://github.com/nashory/gans-collection.torch), torch7
+ [hwalsuklee/tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections), tensorflow
+ [wiseodd/generative-models](https://github.com/wiseodd/generative-models), both pytorch and tensorflow
+ [aboev/arae-tf](https://github.com/aboev/arae-tf), tensorflow

<sub><a href="#contents">↑ back to Contents</a></sub>

___

## Trendy AI-application Articles
+ [Artificial intelligence can say yes to the dress](https://qz.com/1090267/artificial-intelligence-can-now-show-you-how-those-pants-will-fit/)

<sub><a href="#contents">↑ back to Contents</a></sub>

-----

## Author
Minchul Shin, [@nashory](https://github.com/nashory)

__Any recommendations to add to the list are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and feel free to make pull requests! :)__

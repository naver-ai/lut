## Learning with Unmasked Tokens Drives Stronger Vision Learners | [[ECCV 2024]](https://arxiv.org/abs/2310.13593v2)

[Taekyung Kim*](https://scholar.google.co.kr/citations?user=u-9bdkwAAAAJ&hl=en), [Sanghyuk Chun](https://sanghyukchun.github.io/home), [Byeongho Heo](https://sites.google.com/view/byeongho-heo/home), [Dongyoon Han*](https://sites.google.com/site/dyhan0920/) <br>
<sub> (*equal contribution) <br>

[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic/ai-lab)

Official PyTorch implementation of LUT "Learning with Unmasked Tokens Drives Stronger Vision Learners" | [arxiv](https://arxiv.org/abs/2310.13593v3).
<br>

### Abstract

> Masked image modeling (MIM) has become a leading self-supervised learning strategy. MIMs such as Masked Autoencoder (MAE) learn strong representations by randomly masking input tokens for the encoder to process, with the decoder reconstructing the masked tokens to the input. However, MIM pre-trained encoders often exhibit a limited attention span, attributed to MIM’s sole focus on regressing masked tokens only, which may impede the encoder’s broader context learning. To tackle the limitation, we improve MIM by explicitly incorporating unmasked tokens into the training process. Specifically, our method enables the encoder to learn from broader context supervision, allowing unmasked tokens to experience broader contexts while the decoder reconstructs masked tokens. Thus, the encoded unmasked tokens are equipped with extensive contextual information, empowering masked tokens to leverage the enhanced unmasked tokens for MIM. As a result, our simple remedy trains more discriminative representations revealed by achieving 84.2% top-1 accuracy with ViT-B on ImageNet-1K with 0.6%p gain. We attribute the success to the enhanced pre-training method, as evidenced by the singular value spectrum and attention analyses. Finally, our models achieve significant performance gains at the downstream semantic segmentation and fine-grained visual classification tasks; and on diverse robust evaluation metrics
  
![framework](assets/framework.png)
*<p align="center">Framework overview</p>*
  
## Updates
  * (07/2024) LUT is accepted at ECCV 2024 
  
## Preparation
  * This repo is based on [MAE](https://github.com/facebookresearch/mae/tree/main)
  * This repo uses timm==0.4.12 and pytorch==1.13.0
  * pip install -r requirements.txt
  
  
## Training
  For training commands, please refer to [PRETRAIN.md](docs/PRETRAIN.md) and [TRAINING.md](docs/FINETUNE.md).

## Performances
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">ViT-S</th>
<th valign="bottom">ViT-B</th>
<th valign="bottom">ViT-L</th>
<!-- TABLE BODY -->
<tr><td align="left">MoCo v3</td>
<td align="center">81.4</td>
<td align="center">83.2</td>
<td align="center">84.1</td>
<tr><td align="left">DINO</td>
<td align="center">81.5</td>
<td align="center">82.8</td>
<td align="center"> - </td>
<tr><td align="left">iBOT</td>
<td align="center">82.04</td>
<td align="center">84.0</td>
<td align="center">84.8</td>
<tr><td align="left">MAE</td>
<td align="center">81.4</td>
<td align="center">83.7</td>
<td align="center">85.6</td>
<tr><td colspan="4"></td></tr>
<tr><td align="left"> LUT </td>
<td align="center">82.0</td>
<td align="center">84.2</td>
<td align="center">86.0</td>
</tbody></table>


## Citation
```
@article{kim2023lut,
  title={Learning with Unmasked Tokens Drives Stronger Vision Learners},
  author={Kim, Taekyung and Chun, Sanghyuk and Heo, Byeongho and Han, Dongyoon},
  journal={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
  
## License
```
LUT
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)  
```

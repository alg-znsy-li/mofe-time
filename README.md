# mofe-time

## 摘要
在过去的几十年，时间序列预测技术取得了巨大的技术进步，特别是基于预训练方法的时间序列预测技术发展非常迅速。这些技术主要包括LLM4TS和TSLLM两个技术方向，研究认为时间序列是一种可以学习的语言——用LLM辅助或者直接建模时间序列。然而时间序列本质上是一个离散的信号，现有的上述两种建模思路，并没有充分考虑时间序列的信号本征属性。为此，我们提出了一种基于MoE架构的频域学习的基础时序大模型——MoFE-Time。这种技术引入了最新的信号学习网络FAN（傅里叶分析网络）和领先的LLM基础架构MoE，创新性地将频域专家引入MoE架构中，利用MoE的路由算法，使得算法可以关注不同的频域属性，从而更好的建模时间序列的频域本征属性。我们发现相比目前时域最好的开源大模型：（1）在多个不同的数据集中，MoFE-Time模型几乎都具有更好的性能表现。（2）达到同等的零样本预测效果时，MoFE-Time具有更低的训练成本和更快的推理速度；（3）MoFE-Time 具有更好的后训练特性, 微调后模型性能在多个公开数据集上均大幅领先现有的时序模型。 据我们所知，我们的工作是首次在时间序列基础大模型中，结合MoE和信号频域本征属性学习的技术。

## 技术原理
paper: https://arxiv.org/abs/2507.06502

<img width="553" alt="image" src="https://github.com/user-attachments/assets/c3ffe966-2556-4340-8d0b-a5653ad95eb0" />

## 主要技术指标
### zero-shot


### full-shot
<img width="461" alt="image" src="https://github.com/user-attachments/assets/410fe74b-06cd-4a5c-a43f-fe39f95e17a8" />

## Usage
### pretrain
1. data prepare
   
   ![image](https://github.com/user-attachments/assets/4b17181a-6ff7-480f-b66a-e4116d615c04)

   download from huggingface
   https://huggingface.co/datasets/Maple728/Time-300B
3. start pretrain
    ```./```

### fine tune

### infer

## Citation

> 🙋 Please let us know if you find out a mistake or have any suggestions!

> 🌟 If you find the MOFE-Time models helpful in your research, please consider to star this repository and cite the
> corresponding

```
@misc{liu2025mofetimemixturefrequencydomain,
      title={MoFE-Time: Mixture of Frequency Domain Experts for Time-Series Forecasting Models}, 
      author={Yiwen Liu and Chenyu Zhang and Junjie Song and Siqi Chen and Sun Yin and Zihan Wang and Lingming Zeng and Yuji Cao and Junming Jiao},
      year={2025},
      eprint={2507.06502},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.06502}, 
}
```

## Related Resources
* TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis, in arXiv 2024. [\[paper\]](https://arxiv.org/abs/2410.16032) [\[GitHub Repo\]](https://github.com/kwuking/TimeMixer)
* Towards Neural Scaling Laws for Time Series Foundation Models, arXiv 2024. [\[paper\]](https://arxiv.org/pdf/2410.12360)
* Foundation Models for Time Series Analysis: A Tutorial and Survey, in *KDD*
  2024. [\[paper\]](https://arxiv.org/abs/2403.14735) [\[Tutorial\]](https://wenhaomin.github.io/FM4TS.github.io/)
* What Can Large Language Models Tell Us about Time Series Analysis, in *ICML*
  2024. [\[paper\]](https://arxiv.org/abs/2402.02713)
* Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects, in *TPAMI*
  2024. [\[paper\]](https://arxiv.org/abs/2306.10125) [\[Website\]](https://github.com/qingsongedu/Awesome-SSL4TS)
* Transformers in Time Series: A Survey, in *IJCAI*
  2023. [\[paper\]](https://arxiv.org/abs/2202.07125) [\[GitHub Repo\]](https://github.com/qingsongedu/time-series-transformers-review)
* A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection, in *TPAMI* 2024. [\[paper\]](https://arxiv.org/abs/2307.03759) [\[Website\]](https://github.com/KimMeen/Awesome-GNN4TS)



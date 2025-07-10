<div align="center">
  <h2><b>预测基座大模型 MoFE-Time: Mixture of Frequency Domain Experts for Time-Series Forecasting Models
 </b></h2>
</div>

<div align="center">


</div>

<div align="center">

**[<a href="https://arxiv.org/abs/2507.06502">Paper Page</a>]**


</div>

## Abstract
As a prominent data modality task, time series forecasting plays a pivotal role in diverse applications. With the remarkable advancements in Large Language Models (LLMs), the adoption of LLMs as the foundational architecture for time series modeling has gained significant attention. Although existing models achieve some success, they rarely both model time and frequency characteristics in a pretraining-finetuning paradigm leading to suboptimal performance in predictions of complex time series, which requires both modeling periodicity and prior pattern knowledge of signals. We propose MoFE-Time, an innovative time series forecasting model that integrates time and frequency domain features within a Mixture of Experts (MoE) network. Moreover, we use the pretraining-finetuning paradigm as our training framework to effectively transfer prior pattern knowledge across pretraining and finetuning datasets with different periodicity distributions. Our method introduces both frequency and time cells as experts after attention modules and leverages the MoE routing mechanism to construct multidimensional sparse representations of input signals. In experiments on six public benchmarks, MoFE-Time has achieved new state-of-the-art performance, reducing MSE and MAE by 6.95% and 6.02% compared to the representative methods Time-MoE. Beyond the existing evaluation benchmarks, we have developed a proprietary dataset, NEV-sales, derived from real-world business scenarios. Our method achieves outstanding results on this dataset, underscoring the effectiveness of the MoFE-Time model in practical commercial applications.

checkpoint will release soon ^-^
## Paper
paper: https://arxiv.org/abs/2507.06502

<img width="553" alt="image" src="https://github.com/user-attachments/assets/c3ffe966-2556-4340-8d0b-a5653ad95eb0" />

## Main Results
Experimental results demonstrate that our method achieve
SOTA on six public datasets and NEV-sales, which demonstrates the effectiveness of the MoFE-Time model.
### FineTune Results for Six Public Datasets
<p align="center">
<img src=".\figures\main_result.png" width = "800" height = "" alt="" align=center />

### FineTune Results for Proprietary Dataset NEV-sales
<img src=".\figures\nev_sales.png" width = "800" height = "" alt="" align=center />

### Ablation Study
<img src=".\figures\ablation.png" width = "800" height = "" alt="" align=center />
</p>

## Usage
### pretrain
1. data prepare
   
   ![image](https://github.com/user-attachments/assets/4b17181a-6ff7-480f-b66a-e4116d615c04)

   download from huggingface
   https://huggingface.co/datasets/Maple728/Time-300B
2. Install Pytorch and other dependencies.
   ```
   pip install -r requirements.txt
   ```
3. start pretrain
   
    ```sh ./src/pretrain_and_eval_ds.sh```
   
   Pretrain on Multiple Nodes
    ```sh ./src/pretrain_and_eval_nodes.sh```
### fine tune

    ```sh ./src/fine_tune_ds.sh```


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

## Acknowledgement
We appreciate the following GitHub repos a lot for their valuable code and efforts.

Time-MoE (https://github.com/Time-MoE/Time-MoE)

Time-Series-Library （https://github.com/thuml/Time-Series-Library）

Chronos (https://github.com/amazon-science/chronos-forecasting)

Times-FM (https://github.com/google-research/timesfm)

Moirai(https://github.com/SalesforceAIResearch/uni2ts)

## Concat 
If you have any questions or want to use the code, please contact caoyuji@lixiang.com or liuyiwen@lixiang.com 

## Other Work
asLLR: LLM Based Leads Raking In Auto Sales（[https://github.com/alg-znsy-li/as_llr](https://github.com/alg-znsy-li/as_llr)）


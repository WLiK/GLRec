# GLRec
The code of AAAI'24 paper GLRec. [Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations](https://arxiv.org/abs/2307.05722)

This paper focuses on unveiling the capability of large language models in understanding graph data and leveraging this understanding to enhance recommendations.

Due to business privacy issues, we are currently unable to provide commercial data and require further follow-up in accordance with company policies. The real case of training and testing instruction data can be referred to our paper's case illustration. The code of GLRec model has been uploaded to this project. The project will be supplemented and optimized continuously.

Training the model,
```
deepspeed --num_gpus=2 model_GLRec.py --deepspeed configs/ds_zero2.json
```

Our project is developed based on the projects below, thanks for their contributions.

TALLrec: https://github.com/SAI990323/TALLRec

Alpaca_lora: https://github.com/tloen/alpaca-lora

BELLE: https://github.com/LianjiaTech/BELLE

If our work has been of assistance to you, please feel free to cite our paper. Thank you.
```
@article{wu2023exploring,
  title={Exploring large language model for graph data understanding in online job recommendations},
  author={Wu, Likang and Qiu, Zhaopeng and Zheng, Zhi and Zhu, Hengshu and Chen, Enhong},
  journal={arXiv preprint arXiv:2307.05722},
  year={2023}
}
```

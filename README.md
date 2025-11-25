<p align="center">
  <img src="assets/common/banners/model_compression_banner.png" alt="Model Compression Study Banner" width="100%">
</p>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![GitHub Stars](https://img.shields.io/github/stars/TheodorePTP/model_compression_study?style=social)
### 📚 仓库介绍
本仓库用于系统梳理**模型压缩**领域的经典与前沿方法。包括**技术文档**与**项目实践**两部分  
#### 技术分类主要涵盖：
- 🔹 **模型量化（Quantization）**
- ✂️ **模型剪枝（Pruning）**
- 🧪 **知识蒸馏（Knowledge Distillation）**
- ⚡ **部署优化（deployment_optimization）**

### 📁 仓库主要结构
```bash
model_compression_study/
├── docs/                 # 技术文档
│   ├── quantization/     
│   ├── pruning/          
│   └── distillation/   
│   └── deployment_optimization/ 
│
├── experiments/          # 实践代码
│   ├── quantization/      
│   ├── pruning/        
│   └── distillation/     
│   └── deployment_optimization/ 
│
└── README.md             # 项目介绍与使用说明
```
### 💻 更新进度
|docs|进度|文章链接|相关代码|Paper|
|---|-----|--|---|---|
|**deployment_optimization** | 🔴待开始 |\ |\ |
|**pruning** | 🔴待开始 | \ |\ |
|**quantization** | 🔴待开始 | \ |\ |
|**distillation** | 🟢进行中 | [经典论文梳理](docs/distillation/classic_methods.md) |\ |
| KD | 完成✅  | [KD方法解析](https://blog.csdn.net/qq_44923064/article/details/155098435?fromshare=blogdetail&sharetype=blogdetail&sharerId=155098435&sharerefer=PC&sharesource=qq_44923064&sharefrom=from_link) | [KD相关代码](experiments/distillation/KD)|[arXiv:1503.02531](https://arxiv.org/abs/1503.02531) |
|AT|完成✅| [AT方法解析](https://blog.csdn.net/qq_44923064/article/details/155104865?fromshare=blogdetail&sharetype=blogdetail&sharerId=155104865&sharerefer=PC&sharesource=qq_44923064&sharefrom=from_link) | [AT相关代码](experiments/distillation/AT/at_attention_comparison.py) |[arXiv:1612.03928](https://arxiv.org/abs/1612.03928) |
|CRD|待完成| \ | \ | [arXiv:1910.10699](https://arxiv.org/abs/1910.10699) |
|DKD|待完成| \ |\ |[arXiv:2203.08679](https://arxiv.org/abs/2203.08679) |
|FitNet|完成✅| [FitNet方法解析](https://blog.csdn.net/qq_44923064/article/details/155197934?fromshare=blogdetail&sharetype=blogdetail&sharerId=155197934&sharerefer=PC&sharesource=qq_44923064&sharefrom=from_link) |[FitNet相关代码](experiments/distillation/FitNet/generate_fitnet_core_diagram.py) |[arXiv:1412.6550](https://arxiv.org/abs/1412.6550) |
|KDSVD|待完成| \ |\ |[arXiv:1807.06819](https://arxiv.org/abs/1807.06819) |
|NST|待完成| \ |\ |[arXiv:1707.01219](https://arxiv.org/abs/1707.01219) |
|OFD|待完成| \ |\ |[arXiv:1904.01866](https://arxiv.org/abs/1904.01866) |
|PKT|待完成| \ |\ |[arXiv:1803.10837](https://arxiv.org/abs/1803.10837) |
|ReviewkD|完善中| \ |\ |[arXiv:2104.09044](https://arxiv.org/abs/2104.09044) |
|RKD|待完成| \ |\ | [arXiv:1904.05068](https://arxiv.org/abs/1904.05068) |
|SP|待完成| \ |\ | [arXiv:1907.09682](https://arxiv.org/abs/1907.09682) |
|VID|待完成| \ |\ |[arXiv:1904.05835](https://arxiv.org/abs/1904.05835) |


#### ⭐ 如果你觉得本项目有帮助，欢迎点个 Star！
[![Star History Chart](https://api.star-history.com/svg?repos=TheodorePTP/model_compression_action&type=Date)](https://star-history.com/#TheodorePTP/model_compression_action&Date)
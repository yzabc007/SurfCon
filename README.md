# SurfCon
Implementation of SurfCon model for Paper "[SurfCon: Synonym Discovery on Privacy-Aware Clinical Data](https://github.com/yzabc007/SurfCon/)", which studies Synonym Discovery on Privacy-Aware Clinical Data.

## 1. Introduction
This repository is the implementation for the SurfCon model which utilizes the surface form and global context information to mine synonyms for medical terms extracted from Electronic Medical Records (EMRs).

The surface form information provides connection between medical terms in the surface form level. Not surprisingly, surface form information is very important in medical domain. For example, term **hypocupremia** is the synonym of **copper deficiency** in which _hypo_ means _deficiency_, _cupre_ means _copper_ and also _mia_ is connected with _blood_. Thus, we design a bi-level surface form encoder to capture the information in both character and word levels of the medical terms.

The global context information provides the semantic information between medical terms. 

## 2. Dataset
The dataset used in current experiments contains medical term-term co-occurrence graphs extracted from EMRs. The dataset can be downloaded from the original paper, [Building the graph of medicine from millions of clinical narratives](https://datadryad.org/resource/doi:10.5061/dryad.jp917)

There are some inputs that need to be prepared before running. 

More importantly, you can apply our model to your own data. Our model and problem setting can be extended to any other text corpus with the privacy concerns as long as a co-occurrence graph is provided.

## 3. Run

Testing our pretrained SurfCon model to discover synonyms:

> bash test_surfcon.sh

Training the final ranking model:

> bash train_surfcon.sh

Training the inductive context prediction model:

> bash train_ctx_pred.sh


## 4. Citation
```
@inproceedings{wang2019surfcon,
  title={SurfCon: Synonym Discovery on Privacy-Aware Clinical Data},
  author={Wang, Zhen and Yue, Xiang and Moosavinasab, Soheil and Huang, Yungui and Lin, Simon and Sun, Huan},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2019},
  organization={ACM}
}
```

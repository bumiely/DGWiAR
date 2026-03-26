# DGWiAR
Lightweight WiFi-CSI action recognition under unknown target domains.
## Project Structure
This repository is organized into two folders:

### Core Training
Core implementation of DGWiAR, including model definition, backbone networks, dataloaders, CORAL alignment, and training utilities.  
Main files:  
* `DIFEX.py`
* `alg.py`
* `img_network.py`  
* `modelopera.py`  
* `CORAL.py`  
* `opt.py`  

### Data Generation Preprocessing
Scripts for CSI sample generation and preprocessing, including image transformation, RP-related processing, and domain split.  
Main files:  
* `cross_ants_subtrans.py`  
* `trans_image.py`  
* `data_split_domain.py`  
* `RP.py`  


## Method Overview

DGWiAR is built on:  
multi-source domain generalization  
teacher-student distillation  
CORAL-based feature alignment  
feature difference constraint  
lightweight inference with MobileNetV2  

## Install Dependencies  
```bash
pip install -r requirements.txt
```  
##  Data  
You can download the public dataset from [widar3.0](http://tns.thss.tsinghua.edu.cn/widar3.0/)

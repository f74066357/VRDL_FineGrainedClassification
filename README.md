# VRDL_FineGrainedClassification
---
## Dataset:
 6,033 bird images belonging to 200 bird species
 - 3000 training images
 - 3033 testing images 
 
Download Link: https://drive.google.com/file/d/1mDNBHSioQdl-bSxkm5ZDfMrG0uxlZdqg/view?usp=sharing
 
## Create environment
    conda create --name YOURNAME python=3.7.
    conda activate YOURNAME
    conda install tensorflow-gpu==2.0.0
    conda install matplotlib==3.4.3
    pip3 install opencv-python==4.1.1.26
    pip3 install h5py==2.10


## My pretrained model 

Download Link: https://drive.google.com/file/d/1JClNRnIEMgT9qQqCeGBS68aSoNSynzIp/view?usp=sharing

## Inference
    python inference.py  
    
   Remember to put the download link in correct path.  
   DEFAULT path : model/BCNN_keras/1031_7/E[35]_LOS[1.051]_ACC[0.752].h5  
   Or, you can modify line 533 in inference.py to change the default path.
   
## Result
Data      | Loss  | Accuracy | 
------------  | ----  | ---  | 
Training set   | 0.017327397 |  0.9975 | 
Valid set  | 1.050688473 |  0.75166667 |  
   
## Reference:
   - https://github.com/tkhs3/BCNN_keras
   - https://reurl.cc/q1pO2q

## Others:
 - Python code written with PEP8 guidelines

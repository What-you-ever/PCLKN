# PCLKN
This repository is an official implementation of the paper "Adaptive Pixel Classification and Equivalent Large Kernels for Lightweight Image Super-Resolution".

**This GitHub account is anonymous, with no personal information disclosed, and adheres to the double-blind principle.**

## 📊 Results
### Quantitative Comparison

<img src="images/comparisons.png">

<img src="images/bubble_plot.png">

### Visual Comparisons

<img src="images/visual.png">
<img src="images/visual2.png">

## Training & Testing 

### Dependencies 

```bash
git clone https://github.com/What-you-ever/PCLKN.git

conda create -n PCLKN python=3.9
conda activate PCLKN

pip install -r requirements.txt
```

### Train

```bash
# x2 
python basicsr/test.py -opt options/test/PCLKNSR_x2.yml
# x3
python basicsr/test.py -opt options/test/PCLKNSR_x2.yml
# x4
python basicsr/test.py -opt options/test/PCLKNSR_x4.yml
```
### Test

```bash
# x2
python basicsr/test.py -opt options/test/PCLKNSR_x2.yml
# x3
python basicsr/test.py -opt options/test/PCLKNSR_x2.yml
# x4
python basicsr/test.py -opt options/test/PCLKNSR_x4.yml
```


## 🏅 Acknowledgements

This project is built on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary). Special thanks to their excellent works!

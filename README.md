# Intelligent Beam Optimization for Light-Sheet Fluorescence Microscopy through Deep Learning
Here we report that a joint-optimization continuously updates the phase mask and resulting in improved image quality for better cell detection. Our method's efficacy is demonstrated through both simulations and experiments, revealing substantial enhancements in imaging quality compared to traditional Gaussian light sheet. We discuss how designing microscopy systems through a computational approach provides novel insights for advancing optical design that relies on deep learning models for the analysis of imaging datasets.  
Paper reference (doi: https://doi.org/10.1101/2023.11.29.569329)
## Autofocus pipeline
<img src="images/Fig1.png" width="630" height="534">  
## Required packages
pytorch (>= 1.12), torchvision, matplotlib, numpy, random, skimage, scipy
## Function of scipts

**mask_learning.py**

**cnn_utils.py**

**loss_utils.py**
 - loss function design, we deprecate it since we only use cross entropy for simplicity. this part could be future work.
   
**physics_utils.py**

**data_utils.py**

**Generate_data.py**

**psf_gen.py**

**test_localization.py**

**submit.sh**
 - bash file for the cluster training

## Contact
cli38@ncsu.edu

## References
- Li, Chen, et al. "Deep learning-based autofocus method enhances image quality in light-sheet fluorescence microscopy." Biomedical Optics Express 12.8 (2021): 5214-5226.
- Royer, Loïc A., et al. "Adaptive light-sheet microscopy for long-term, high-resolution imaging in living organisms." Nature biotechnology 34.12 (2016): 1267-1278.
- 

# OASIS2-CNN


Dementia is a threatening condition that affects communication, thinking, and memory skills, being Alzheimer its most
common type. The early detection of this disease allows for better care of the patient. Recently, Machine Learning (ML)
methods have been developed to support the finding and forecast of Alzheimer’s disease through the analysis of Magnetic
Resonance Images (MRI). Existing ML methods present some limitations: (i) require an expert to extract relevant features
from MRI, (ii) depend on multistep image preprocessing, or (iii) need complex architectures and several images to train
them. To surpass these limitations, in the present work, we analyze different Convolutional Neural Networks (CNNs) for
Alzheimer’s classification, formulated to learn from a set of representative MRI sagittal images available in the Open
Access Series of Imaging Studies (OASIS-2, 72 non-demented and 64 demented subjects, with ages from 60 to 96 years)
and the Alzheimer’s Disease Neuroimaging Initiative (ADNI, 200 early Alzheimer and 200 control patients, with ages from
55 to 90 years) datasets. All CNNs were compared with state-of-the-art ML methods, being the VGG-16 variant the best
performed architecture with an average validation accuracy of 56% ± 4%, evaluated with a bootstrapping strategy to
measure the variability on independent runs. This result confirms the best performance reported so far (\60%) with
different ML methods. The low accuracy evidences the hardness of the problem and contrasts with the higher accuracy
levels (up to 97%) reached with preprocessed and well-characterized MRI axial images from the OASIS-1 or ADNI-2
datasets. Thus, opening an interesting discussion about what MRI plane should be considered when training CNNs for
Alzheimer’s classification, and leaving a wide room for improvement on the performance of CNNs trained with sagittal
MRI images. The resulting model implemented in software and experimental data are publicly available.

Cite this article
Waldo-Benítez, G., Padierna, L.C., Ceron, P. et al. Dementia classification from magnetic resonance images by machine learning. Neural Comput & Applic (2023). https://doi.org/10.1007/s00521-023-09163-y

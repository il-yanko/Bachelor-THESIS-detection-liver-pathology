# Texture Analysis of Medical Ultrasound images (Liver)

This is my bachelor thesis project at Igor Sikorsky Kyiv Polytechnic Institute.

The project is a classification system for pathologies of the hepatobiliary system in children, based on the analysis of their liver ultrasound images using radiomics (computational methods for texture analysis) and machine learning.

### Feature Extracting Techniques:
- [x] GLCM (Gray-Level Co-Occurrence Matrix)
- [x] GLRLM (Gray-Level Long Run Matrix)
- [x] GLDM (Gray-Level Dependence Matrix)
- [x] NGLDM (Neighborhood Gray-Level Different Matrix)
- [x] GLSZM (Gray-Level Size Zone Matrix)
- [ ] GLAM (Gray-Level  Aura  Matrix)
- [ ] GLNM (Gray-Level  Neighbor  Matrix)
- [x] Principal Component Analysis

### Classification Approaches:
- [x] Decision Tree Classifier
- [x] Logistic Regression (logit)
- [x] Random Forest Classifier
- [x] C-Support Vector Machine
- [x] Gradient Boosting
- [x] k-Nearest Neighbors
- [x] Multi-layer Perceptron

### Classificator Estimator:
- [x] 5-fold Cross-Validation


## References ##

1. Haralick R. M. Textural features for image classification / R. M. Haralick, K. S. Shanmugam, I. Dinstein // IEEE Transactions on Systems, Man, and Cybernetics. — 1973. — P. 610–621.
2. Galloway M. M. Texture analysis using gray level run lengths / M. M. Galloway // Computer Graphics and Image Processing. — 1974. — Vol. 4, No. 2. — P. 172–179.
3. Thibault G. Texture indexes and gray level size zone matrix application to cell nuclei classification / G. Thibault, B. Fertil, C. Navarro, S. Pereira // Pattern Recognition and Information Processing. — 2009. — No. November. — P. 140–145.
4. Hosny A. Computational radiomics system to decode the radiographic phenotype / A. Hosny, J. J. M. van Griethuysen, C. Parmar [et al.] // Cancer Research. — 2017. — Vol. 77, No. 21. — P. e104–e107.

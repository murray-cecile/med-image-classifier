
#### pre-processing
- top % of gray values relative to image
- same values for testing as training, make sure the normalization is the same

#### segmentation
- IOU = A intersect T / A union T
- DICE = 2 A intersect T / A + T
- **EVERYTHING IS TASK-BASED**
- try seeing what happens when we use the truth!

#### feature generation
- best spiculation: look for a Giger paper from the 1980s
    - outlining not super helpful
    - analyze the patterns of lines across the image
    - need to distinguish between the circle and the blob
    - find a maximum gradient to the center of the region: greater spread is greater spiculation
    - did it relative to the segmentation, did it within a band, and did it within the whole ROI: go along whole image because the whole image is ROI
    - edge enhancement
    - analyze the outside (the negative image)
- irregularity
- **simple texture measure is standard deviation**
- three merged low level classifiers will do better

#### classification
- try SVM and LDA
- try running ROC on one feature only to make sure we're not flipping the labels
- be careful when we bounce around 

#### evaluation
- they do roc because it's not dependent on the 
- optimal cutoff on ROC curve: euclidean average on sensitivity and specificity
- partial AUCs: specify whether you care more about sensitivity or specificity, given same ROC

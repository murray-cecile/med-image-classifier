#!usr/bin/bash

# PROVIDE THE FULL FILEPATH TO THE DIRECTORY WHERE IMAGES ARE STORED AS A COMMAND LINE ARGUMENT
# RUN THIS IN THE DIRECTORY WHERE THE IMAGES ARE SAVED


# Step 1: Add desired images to the cart and download (I bulk selected the first 100)
# Step 2: use NBIA data retriever to convert images from .tcia to .dcm (longest step)

# Step 3: Flatten downloaded folders using the following linux command:
FILEDIR=$1
find $FILEDIR -type f -exec sh -c 'for f do x=${f#./}; y="${x// /_}"; eval "mv ${x// /\ } ${y////-}"; done' {} +

# Step 4: remove filepath portion of directory name from all files using the following Linux command:
cd $FILEDIR
rename -- "s/-.+-med-image-classifier-raw_train-//" *
rename -- "s/-.+-med-image-classifier-raw_test--CBIS-DDSM-//" *
# this works for Tammy
# rename -- "s/raw_test--CBIS-DDSM-//" *

# Step 5: optional, move the binary ROI masks to a different subdirectory
mkdir binary_roi
find . -type f -size +1M -exec mv {} binary_roi \;
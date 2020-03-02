#!usr/bin/bash

cat metadata/mass_case_description_train_set.csv | grep -o Mass-Training_P_\\d\\+ | sort -u > metadata/mass_train_ids.csv
cat metadata/mass_case_description_test_set.csv | grep -o Mass-Testing_P_\\d\\+ | sort -u > metadata/mass_test_ids.csv

sed -i .bak '$!s/$/,/' metadata/mass_train_ids.csv 
sed -i .bak '$!s/$/,/' metadata/mass_test_ids.csv 
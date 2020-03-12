#===============================================================================#
# CREATE DATA VIZ FOR CV PROJECT
#
# Cecile Murray
#===============================================================================#

setwd("/Users/cecilemurray/Documents/CAPP/MPHY-39600/med-image-classifier")

libs <- c("here",
          "tidyverse",
          "magrittr",
          "purrr",
          "knitr", 
          "kableExtra",
          "janitor",
          "rstanarm")
lapply(libs, library, character.only = TRUE)

# read in data
train100 <- read_csv("final/current_train.csv")
test100 <- read_csv("final/current_test.csv")
full_train <- read_csv("final/full_current_train.csv")
full_test <- read_csv("final/full_current_test.csv")

#===============================================================================#
# EXPLORE SPICULATION
#===============================================================================#

# X1"                  "area"                "circularity"         "convex_area"        
#  [5] "eccentricity"        "equivalent_diameter" "gabor"               "hough"              
#  [9] "id"                  "iou"                 "major_axis_length"   "minor_axis_length"  
# [13] "perimeter"           "snake"               "spiculationA"        "spiculationB"       
# [17] "spiculationC"        "spiculationD"        "spiculationRA"       "spiculationRB"      
# [21] "spiculationRC"       "spiculationRD"       "pathology"        

# histograms
test100 %>% 
  mutate(label = if_else(pathology == 0, "Benign", "Malignant")) %>% 
  ggplot(aes(x = perimeter, 
             fill = label)) +
  geom_histogram() +
  scale_fill_manual(values = c("Malignant" = "red", "Benign" = "green")) +
  facet_wrap(~label, labeller = "label_both",
             nrow = 2) +
  labs(title = "Perimeter distribution by pathology",
       x = "Perimeter",
       y = "Count",
    fill = "") +
  theme_minimal() +
  theme(legend.position = c(0.8, 0.9))

full_train %>% 
  ggplot(aes(x = spiculationB,
             y = spiculationD,
             color = as.factor(pathology))) +
  geom_point() +
  scale_color_manual(values = c("1" = "red", "0" = "green"),
                     labels = c("Malignant", "Benign")) +
  labs(title = "",
       color = "") +
  theme_minimal() +
  theme(legend.position = c(0.1, 0.9)) 

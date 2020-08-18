# set the working directory
setwd("C:/Users/96171/Desktop/ministry_of_public_health/Code")

# read the training and testing data and combine them into one
df_train <- read.csv(file = '../input/all_without_date/collated/all_columns/df_train_collated.csv')
df_test <- read.csv(file = '../input/all_without_date/collated/all_columns/df_test_collated.csv')
df <- rbind(df_train, df_test)

#install.packages("devtools")
library("devtools")
#install_github("vqv/ggbiplot")
library(ggbiplot)

# create a vector were values are either 'high' or 'low'
demand <- df$demand
demand_type <- ifelse(demand < 250, "low demand", "high demand")

# rename columns to make representation shorter
library(plyr)
df <- rename(df, c("demand" = "demand",
            "civilians_rank" = "civ",
            "distance" = "dist",
            "AverageTemp" = "AT",
            "AverageWindSpeed" = "AWS",
            "Precipitation" = "precp",
            "w_.t.1." = "t1",
            "w_.t.2." = "t2",
            "w_.t.3." = "t3",
            "w_.t.4." = "t4",
            "w_.t.5." = "t5",
            "w_.t.1._trend" = "tnd",
            "w_.t.1._seasonality" = "ssn",
            "service_General.Medicine" = "GMed",
            "service_Gynaecology" = "Gyn",
            "service_Pediatrics" = "Ped",
            "service_Pharmacy" = "Phrm",
            "mohafaza_B" = "mB",
            "mohafaza_N" = "mN",
            "mohafaza_NE" = "mNE"))

# define the pca variable
df.pca <- prcomp(df, center = TRUE,scale. = TRUE)

summary(df.pca)

# pca plot with ellipses and points
png(filename="../output/feature_selection/non_ubr/pca.png")
ggbiplot(df.pca,  obs.scale = 1, var.scale = 1,
         varname.size = 4, varname.adjust=5,
         groups = demand_type, ellipse = TRUE,
         alpha=0.2) + 
  #scale_colour_manual(name="Origin", values= c("forest green", "red3", "dark blue")) +
  ggtitle("PCA")+
  theme_minimal()+
  theme(legend.position = "bottom") 
dev.off()

# pca plot with no ellipses and points 
png(filename="../output/feature_selection/non_ubr/pca_pure.png")
ggbiplot(df.pca, obs.scale = 1, var.scale = 1,
  alpha=0, varname.size = 4, varname.adjust=5) + 
  ggtitle("PCA")+
  theme_minimal()+
  theme(legend.position = "bottom") 
dev.off()
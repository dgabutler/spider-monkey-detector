# Plotting Figures Script

# run from Code directory, contains relative paths
# install.packages("RcppCNPy")
library(RcppCNPy)
library(ggplot2)
library(plyr)
library(ggsignif)

# performance with increasing sample size:

# read in sample size data, add column with standard error
sample_sizes_recall <- as.data.frame(npyLoad('../Results/sample_sizes_RECALL.npy'))
colnames(sample_sizes_recall) <- c("mean_recall","std_recall")
sample_sizes_recall$SE <- sample_sizes_recall$std_recall/sqrt(10)
sample_sizes_recall$size <- c(50, 75, 100, 124)
sample_sizes_recall
sample_sizes_f1 <- as.data.frame(npyLoad('../Results/sample_sizes_F1.npy'))
colnames(sample_sizes_f1) <- c("mean_f1","std_f1")
sample_sizes_f1$SE <- sample_sizes_f1$std_f1/sqrt(10)
sample_sizes_f1$size <- c(50, 75, 100, 124)
sample_sizes_f1

# scatterplot
dev.off()
size_plot_f1 <- ggplot(sample_sizes_f1, aes(x=size, y=mean_f1)) +
    geom_point(shape=1) +    # Use hollow circles
    ylim(40,100) +           # Set Y-axis limits
    ggtitle("Effect of increasing positive sample size on F1 score") +
    xlab("Number of positives in training set") +           # Set X-axis limits
    ylab("Mean F1 score (%)") +           # Set X-axis limits
    geom_smooth(method=lm,   # Add linear regression line
                se=FALSE) +  # Don't add shaded confidence region
    geom_errorbar(aes(ymin=mean_f1-SE, ymax=mean_f1+SE), width=.2,
                position=position_dodge(.9))

print(size_plot_f1)

dev.off()
size_plot_recall <- ggplot(sample_sizes_recall, aes(x=size, y=mean_recall)) +
    geom_point(shape=1) +    # Use hollow circles
    ylim(40,100) +           # Set Y-axis limits
    ggtitle("Effect of increasing positive sample size on recall score") +
    xlab("Number of positives in training set") +           # Set X-axis limits
    ylab("Mean recall score (%)") +           # Set X-axis limits
    geom_smooth(method=lm,   # Add linear regression line
              se=FALSE) +  # Don't add shaded confidence region
    geom_errorbar(aes(ymin=mean_recall-SE, ymax=mean_recall+SE), width=.2,
                position=position_dodge(.9))

print(size_plot_recall)

######################################################################################################

prop_to_perc <- function(x) x * 100

# read in diff. preproc. data
without <- as.data.frame(npyLoad('../Results/WITHOUT.npy'))
colnames(without) <- c("accs","losses","precs","recs","f1s")
without <- data.frame(lapply(without[c("precs","recs","f1s")], prop_to_perc))
without <- stack(without)
without$preprocessing <- "none"
colnames(without)[2] <- "metric"
#
standard <- as.data.frame(npyLoad('../Results/STAND.npy'))
colnames(standard) <- c("accs","losses","precs","recs","f1s")
standard <- data.frame(lapply(standard[c("precs","recs","f1s")], prop_to_perc))
standard <- stack(standard)
standard$preprocessing <- "standardised"
colnames(standard)[2] <- "metric"
#
denoised <- as.data.frame(npyLoad('../Results/DENOISED.npy'))
colnames(denoised) <- c("accs","losses","precs","recs","f1s")
denoised <- data.frame(lapply(denoised[c("precs","recs","f1s")], prop_to_perc))
denoised <- stack(denoised)
denoised$preprocessing <- "denoised"
colnames(denoised)[2] <- "metric"
#
denoised_standard <- as.data.frame(npyLoad('../Results/DENOISED_STAND.npy'))
colnames(denoised_standard) <- c("accs","losses","precs","recs","f1s")
denoised_standard <- data.frame(lapply(denoised_standard[c("precs","recs","f1s")], prop_to_perc))
denoised_standard <- stack(denoised_standard)
denoised_standard$preprocessing <- "denoised_standard"
colnames(denoised_standard)[2] <- "metric"
#
aug <- as.data.frame(npyLoad('../Results/ALL_AUG.npy'))
colnames(aug) <- c("accs","losses","precs","recs","f1s")
aug <- data.frame(lapply(aug[c("precs","recs","f1s")], prop_to_perc))
aug <- stack(aug)
aug$preprocessing <- "aug"
colnames(aug)[2] <- "metric"

#### COMPARISON OF PREPROCESSING TECHNIQUES

# PLOTTING:

all_manipulations <- rbind(without,standard,denoised,denoised_standard,aug)
all_manipulations$manipulation <- as.factor(all_manipulations$manipulation)
all_manipulations

# summary stats for stats
summary_manipulations <- ddply(all_manipulations, c("metric", "preprocessing"), summarise,
               N    = length(values),
               mean = mean(values),
               sd   = sd(values),
               se   = sd / sqrt(N))

summary_manipulations$preprocessing <- factor(summary_manipulations$preprocessing, levels=c("none","standardised","denoised","denoised_standard","aug"))

### PLOT
dev.off()
met <-ggplot(summary_manipulations, aes(y=mean, x=metric, color=preprocessing, fill=preprocessing)) +
  ggtitle("Effect of manipulating input data (preprocessing/augmenting) \non various measures of CNN performance") +
  ylab("mean (%)") +      
  geom_bar(position=position_dodge(), stat="identity") + 
  geom_errorbar(aes(ymin=mean-se, ymax=mean+se),
                color="black", 
                width=.2,                    
                position=position_dodge(.9))

met <- met + annotate("text", x=3.365, y=99, label="**", size=6) 

print(met)

# each technique compared to baseline of no method used:

# f1s...
stand_vs_not <- rbind(without,standard)
wilcox.test(f1s ~ preprocessing, data=stand_vs_not) # p-value = 0.2799
#
denoised_vs_not <- rbind(without,denoised)
wilcox.test(f1s ~ preprocessing, data=denoised_vs_not) # p-value = 0.9097
#
denoised_standard_vs_not <- rbind(without,denoised_standard)
wilcox.test(f1s ~ preprocessing, data=denoised_standard_vs_not) # p-value = 0.1986

# accs...
stand_vs_not <- rbind(without,standard)
wilcox.test(accs ~ preprocessing, data=stand_vs_not) # p-value = 0.4256
#
denoised_vs_not <- rbind(without,denoised)
wilcox.test(accs ~ preprocessing, data=denoised_vs_not) # p-value = 0.7615
#
denoised_standard_vs_not <- rbind(without,denoised_standard)
wilcox.test(accs ~ preprocessing, data=denoised_standard_vs_not) # p-value = 0.1111

# losses...
stand_vs_not <- rbind(without,standard)
wilcox.test(losses ~ preprocessing, data=stand_vs_not) # p-value = 0.6305
#
denoised_vs_not <- rbind(without,denoised)
wilcox.test(losses ~ preprocessing, data=denoised_vs_not) # p-value = 0.7959
#
denoised_standard_vs_not <- rbind(without,denoised_standard)
wilcox.test(losses ~ preprocessing, data=denoised_standard_vs_not) # p-value = 0.2799

# precs...
stand_vs_not <- rbind(without,standard)
wilcox.test(precs ~ preprocessing, data=stand_vs_not) # p-value = 0.5288
#
denoised_vs_not <- rbind(without,denoised)
wilcox.test(precs ~ preprocessing, data=denoised_vs_not) # p-value = 0.2567
#
denoised_standard_vs_not <- rbind(without,denoised_standard)
wilcox.test(precs ~ preprocessing, data=denoised_standard_vs_not) # p-value = 0.1655

# recs
stand_vs_not <- rbind(without,standard)
wilcox.test(recs ~ preprocessing, data=stand_vs_not) # p-value = 0.6772
#
denoised_vs_not <- rbind(without,denoised)
wilcox.test(recs ~ preprocessing, data=denoised_vs_not) # p-value = 0.791
#
denoised_standard_vs_not <- rbind(without,denoised_standard)
wilcox.test(recs ~ preprocessing, data=denoised_standard_vs_not) # p-value = 0.7329


#### AUGMENTATION VS. NO AUGMENTATION

aug_vs_not <- rbind(without,aug)

t.test(f1s ~ preprocessing, data=aug_vs_not) # p-value = 0.153
wilcox.test(f1s ~ preprocessing, data=aug_vs_not) # p-value = 0.4359

t.test(accs ~ preprocessing, data=aug_vs_not) # p-value = 0.05155
wilcox.test(accs ~ preprocessing, data=aug_vs_not) # p-value = 0.03051 *

t.test(losses ~ preprocessing, data=aug_vs_not) # p-value = 0.04628 *
wilcox.test(losses ~ preprocessing, data=aug_vs_not) # p-value = 0.315

t.test(precs ~ preprocessing, data=aug_vs_not) # p-value = 0.1782
wilcox.test(precs ~ preprocessing, data=aug_vs_not) # p-value = 0.153

t.test(recs ~ preprocessing, data=aug_vs_not) # p-value = 0.0004226 ***
wilcox.test(recs ~ preprocessing, data=aug_vs_not) # p-value = 0.002765 **



####################################################################################
### GRAVEYARD ##############

# all_recall <- data.frame(without$w_recs,standard$s_recs,denoised$d_recs,denoised_standard$ds_recs,aug$a_recs)

# kruskal wallace
# all_preprocessing <- rbind(without,standard,denoised,denoised_standard)
# all_preprocessing$preprocessing <- as.factor(all_preprocessing$preprocessing)
# 
# f1s <- lm(f1s ~ preprocessing, data=all_preprocessing)
# anova(f1s)                                                     # Pr(>F) = 0.31
# kruskal.test(f1s ~ preprocessing, data=all_preprocessing)      # p-value = 0.3914
# 
# accs <- lm(accs ~ preprocessing, data=all_preprocessing)
# anova(accs)                                                    # Pr(>F) = 0.1791
# kruskal.test(accs ~ preprocessing, data=all_preprocessing)     # p-value = 0.2098
# 
# losses <- lm(losses ~ preprocessing, data=all_preprocessing)
# anova(losses)                                                  # Pr(>F) = 0.4668
# kruskal.test(losses ~ preprocessing, data=all_preprocessing)   # p-value = 0.4762
# 
# precs <- lm(precs ~ preprocessing, data=all_preprocessing)
# anova(precs)                                                   # Pr(>F) = 0.07929
# kruskal.test(precs ~ preprocessing, data=all_preprocessing)    # p-value = 0.09206
# 
# recs <- lm(recs ~ preprocessing, data=all_preprocessing)
# anova(recs)                                                    # Pr(>F) = 0.8149
# kruskal.test(recs ~ preprocessing, data=all_preprocessing)     # p-value = 0.9386

# read in diff. preproc. data - OLD WAY, PRESTACKING FOR PLOTTING
# without <- as.data.frame(npyLoad('../Results/WITHOUT.npy'))
# colnames(without) <- c("accs","losses","precs","recs","f1s")
# without$preprocessing <- "none"
# standard <- as.data.frame(npyLoad('../Results/STAND.npy'))
# colnames(standard) <- c("accs","losses","precs","recs","f1s")
# standard$preprocessing <- "standardised"
# denoised <- as.data.frame(npyLoad('../Results/DENOISED.npy'))
# colnames(denoised) <- c("accs","losses","precs","recs","f1s")
# denoised$preprocessing <- "denoised"
# denoised_standard <- as.data.frame(npyLoad('../Results/DENOISED_STAND.npy'))
# colnames(denoised_standard) <- c("accs","losses","precs","recs","f1s")
# denoised_standard$preprocessing <- "denoised_standardised"
# aug <- as.data.frame(npyLoad('../Results/ALL_AUG.npy'))
# colnames(aug) <- c("accs","losses","precs","recs","f1s")
# aug$preprocessing <- "augmented"

# PREVIOUS TO ADDING AUG TO DATA FRAME

# all_preprocessing <- rbind(without,standard,denoised,denoised_standard)
# all_preprocessing$preprocessing <- as.factor(all_preprocessing$preprocessing)
# all_preprocessing

# read in diff. preproc. data
# without <- as.data.frame(npyLoad('../Results/WITHOUT.npy'))
# colnames(without) <- c("accs","losses","precs","recs","f1s")
# without <- data.frame((lapply(without[c("accs","precs","recs","f1s")], prop_to_perc)),without["losses"])
# without <- stack(without)
# without$preprocessing <- "none"
# colnames(without)[2] <- "metric"
# #
# standard <- as.data.frame(npyLoad('../Results/STAND.npy'))
# colnames(standard) <- c("accs","losses","precs","recs","f1s")
# standard <- data.frame((lapply(standard[c("accs","precs","recs","f1s")], prop_to_perc)),standard["losses"])
# standard <- stack(standard)
# standard$preprocessing <- "standardised"
# colnames(standard)[2] <- "metric"
# #
# denoised <- as.data.frame(npyLoad('../Results/DENOISED.npy'))
# colnames(denoised) <- c("accs","losses","precs","recs","f1s")
# denoised <- data.frame((lapply(denoised[c("accs","precs","recs","f1s")], prop_to_perc)),denoised["losses"])
# denoised <- stack(denoised)
# denoised$preprocessing <- "denoised"
# colnames(denoised)[2] <- "metric"
# #
# denoised_standard <- as.data.frame(npyLoad('../Results/DENOISED_STAND.npy'))
# colnames(denoised_standard) <- c("accs","losses","precs","recs","f1s")
# denoised_standard <- data.frame((lapply(denoised_standard[c("accs","precs","recs","f1s")], prop_to_perc)),denoised_standard["losses"])
# denoised_standard <- stack(denoised_standard)
# denoised_standard$preprocessing <- "denoised_standard"
# colnames(denoised_standard)[2] <- "metric"
# #
# aug <- as.data.frame(npyLoad('../Results/ALL_AUG.npy'))
# colnames(aug) <- c("accs","losses","precs","recs","f1s")
# aug <- data.frame((lapply(aug[c("accs","precs","recs","f1s")], prop_to_perc)),aug["losses"])
# aug <- stack(aug)
# aug$preprocessing <- "aug"
# colnames(aug)[2] <- "metric"

### OTHER VERSION OF PLOT THAT EMPHASISES SLIGHTLY WRONG COMPARISON
# dev.off()
# pre <-ggplot(summary_manipulations, aes(y=mean, x=metric, color=metric, fill=metric)) +
#   geom_bar(position=position_dodge(), stat="identity") +    
#   facet_wrap(~preprocessing) +
#   geom_errorbar(aes(ymin=mean-se, ymax=mean+se),
#                 width=.2,                    # Width of the error bars
#                 position=position_dodge(.9))
# print(pre)
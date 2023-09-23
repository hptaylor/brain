library(mgcv)
library(gamm4)
library(MASS)

args <- commandArgs(trailingOnly = TRUE)
working_directory_path <- args[1]
csv_file_name <- args[2]
kval <- strtoi(args[3])
getMax <- args[4]

# Convert command line string to boolean
convert_to_boolean <- function(input_str) {
  input_str <- tolower(input_str)  # Convert string to lowercase
  if (input_str %in% c("true", "yes", "1")) {
    return(TRUE)
  } else if (input_str %in% c("false", "no", "0")) {
    return(FALSE)
  } else {
    stop("Invalid boolean input!")
  }
}
getMax <- convert_to_boolean(getMax)


#working_directory_path <- "/Users/patricktaylor/lifespan_analysis/individual/t10p_trans/dataframes/metrics/"
#csv_file_name <- "dispersion"
#kval <- "5"
#working_directory_path="/Users/patricktaylor/Documents/lifespan_analysis/individual/t10p_s2_e07/dataframes/metrics/"
#csv_file_name = "dispersion"
setwd(working_directory_path) # path to file
Yhat_file_name <- paste( csv_file_name, "_fit_3M.csv", sep="")
Se_file_name <- paste(csv_file_name, "_standard_error_3M.csv", sep="")
Rsq_file_name <- paste( csv_file_name, "_rsq_3M.csv", sep="")
HDI_file_name <- paste( csv_file_name, "_HDI_3M.csv", sep="")

max_options_vec <- as.logical(read.csv("mr_max_yeo7_options.csv", header=FALSE)$V1)

T <- read.csv(file = paste(csv_file_name,".csv",sep=""))

Nv <- dim(T)[2] - 2  # No. of vertices
Ns <- dim(T)[1]      # No. of samples

ID <- T$Name  # subject ID
age <- T$Age  # age in months


data<-array(0,dim=c(Ns,Nv))

for (k in 1:Nv){
  data[1:Ns,k]<-T[,(k+2)]
}

# ============================= #
#      Fit GAMM to "data"       #
# ============================= #

# ------ Curve Fitting ------ #
maxAge<-max(age) # age in months
minAge<-min(age)
ageAtlas<-c(seq(minAge, (maxAge), by=0.25))
Na<-length(ageAtlas)

transform<-array(0,dim=c(Ns,Nv))
transform<-data

transformData<-array(0,dim=c(Ns,3))
transformData[1:Ns,1]<-factor(t(ID))
transformData[1:Ns,2]<-t(log2(age+1)) # log-transformed age


Yhat<-array(0,dim=c(Na,Nv))
Se<-array(0,dim=c(Na,Nv))
Rsq<-array(0,dim=c(Nv))

# ---- Vertex-wise GAMM fitting ----- #
max_ages <- array(0, dim=c(Nv, 20000))

#Nv=10 ###comment
for (k in 1:Nv){
  getMax <- max_options_vec[k]
  
  print(paste('Param ID:',k))
  transformData[1:Ns,3]<-t(transform[1:Ns,k])
  colnames(transformData)<-c("ID","age","data")
  xfmData.df <- as.data.frame(transformData)
  # GAMM model
  # value for 'k' in s(age,k = 30, bs = 'cs') can be changed
  gamm_mod = gamm4(data~s(age,k = kval, bs = 'cs'),data = xfmData.df,random=~(1|ID),REML=TRUE)
  p<-predict(gamm_mod$gam, data.frame(age = log2(ageAtlas+1)),se.fit = TRUE,interval = "confidence",level = 0.99,type = "response")
  Yhat[1:Na,k]=p$fit
  Se[1:Na,k]=p$se.fit
  Rsq[k]=summary(gamm_mod$gam)$r.sq
  beta_mean <- coef(gamm_mod$gam)
  age_inds <- grep("^s\\(age\\)", names(beta_mean))
  beta_mean <- beta_mean[age_inds]
  beta_vcov <- vcov(gamm_mod$gam)[age_inds, age_inds]
  # 2. Draw samples from the posterior
  betas <- mvrnorm(20000, beta_mean, beta_vcov)
  # 3. Compute the linear predictor matrix for a grid of age values
  Xp <- predict(gamm_mod$gam, newdata = data.frame(age = log2(ageAtlas+1)), type = "lpmatrix")
  Xp <- Xp[, age_inds]
  # 4. Multiply by the drawn samples to get the predicted fits
  fits <- Xp %*% t(betas)
  # 5. Determine the age (from ageAtlas) of the maximum fit for each sample
  if (getMax){
    extreme_rows <- apply(fits, 2, which.max)
  } else {
    extreme_rows <- apply(fits, 2, which.min)
  }
  max_ages[k, ] <- ageAtlas[extreme_rows]
  
}

write.csv(Yhat, Yhat_file_name)
write.csv(Se, Se_file_name)
write.csv(Rsq, Rsq_file_name)

# Compute the HDI for each vertex
library(HDInterval)

HDI_results <- list()

for (k in 1:Nv){
  HDI_results[[k]] <- hdi(max_ages[k, ])
}

# If you want to save the HDI results:
write.csv(as.data.frame(t(sapply(HDI_results, `[`, 1:2))), HDI_file_name)
library(mgcv)
library(gamm4)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Please provide the path to the working directory, the CSV file name, and the prefix for the output CSV files as separate command-line arguments.")
}
working_directory_path <- args[1]
csv_file_name <- args[2]
kval <- strtoi(args[3])
#working_directory_path="/Users/patricktaylor/Documents/lifespan_analysis/individual/t10p_s2_e07/dataframes/metrics/"
#csv_file_name = "dispersion"

Yhat_file_name <- paste( csv_file_name, "_fit_3M.csv", sep="")
Se_file_name <- paste(csv_file_name, "_standard_error_3M.csv", sep="")
Rsq_file_name <- paste( csv_file_name, "_rsq_3M.csv", sep="")

setwd(working_directory_path) # path to file
T <- read.csv(file = paste(csv_file_name,".csv",sep=""))

Nv <- dim(T)[2] - 3  # No. of vertices
Ns <- dim(T)[1]      # No. of samples

ID <- T$Name  # subject ID
age <- T$Age  # age in months
Cohort_ID <- T$Cohort_ID

data<-array(0,dim=c(Ns,Nv))

for (k in 1:Nv){
  data[1:Ns,k]<-T[,(k+3)]
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

transformData<-array(0,dim=c(Ns,4))
transformData[1:Ns,1]<-factor(t(ID))
transformData[1:Ns,2]<-t(log2(age+1)) # log-transformed age
transformData[1:Ns,3]<-factor(t(Cohort_ID))

Yhat<-array(0,dim=c(Na,Nv))
Se<-array(0,dim=c(Na,Nv))
Rsq<-array(0,dim=c(Nv))
# ---- Vertex-wise GAMM fitting ----- #

#Nv=10 ###comment
for (k in 1:Nv){
  print(paste('Param ID:',k))
  transformData[1:Ns,4]<-t(transform[1:Ns,k])
  colnames(transformData)<-c("ID","age","Cohort_ID","data")
  xfmData.df <- as.data.frame(transformData)
  # GAMM model
  # value for 'k' in s(age,k = 30, bs = 'cs') can be changed
  gamm_mod = gamm4(data~s(age,k = kval, bs = 'cs'),data = xfmData.df,random=~(1|ID)+(1|Cohort_ID),REML=TRUE)
  p<-predict(gamm_mod$gam, data.frame(age = log2(ageAtlas+1)),se.fit = TRUE,interval = "confidence",level = 0.99,type = "response")
  Yhat[1:Na,k]=p$fit
  Se[1:Na,k]=p$se.fit
  Rsq[k]=summary(gamm_mod$gam)$r.sq
}
write.csv(Yhat, Yhat_file_name)
write.csv(Se, Se_file_name)
write.csv(Rsq, Rsq_file_name)
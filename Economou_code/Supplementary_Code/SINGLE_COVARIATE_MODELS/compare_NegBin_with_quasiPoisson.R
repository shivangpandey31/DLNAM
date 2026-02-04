### Compare estimates from a Negative Binomial and a 
### quasiPoisson DLNM, fitted as GAMs.

## load R libraries
library(ggplot2)
library(viridis)
library(mgcv)
library(data.table)
library(dlnm)

## The data set
head(chicagoNMMAPS)
## get cleaner version
dat <- data.table(y=chicagoNMMAPS$cvd,temp0=chicagoNMMAPS$temp)
dat


## First, fit the two models
max_lag <- 20
# set up the covariate matrices
n <- nrow(dat)
for(i in 1:max_lag){
  dat[,paste("temp",i,sep="") := c(rep(NA,i),temp0[1:(n-i)]) ]
}
index <- grep("temp", colnames(dat) )
tempDat <- na.omit( dat )
TEMP <- as.matrix( tempDat[,..index] )
LAG <- matrix(0:max_lag,nrow(TEMP),length(0:max_lag),byrow=TRUE) 
W <- matrix(1,length(0:max_lag),nrow=nrow(LAG),ncol=ncol(LAG)) 
# negative binomial
gamDLNM_NB <- gam(y~te(TEMP,LAG,by=W,k=c(10,10)),family=nb,method="REML",data=tempDat) 
# quasi-Poisson
gamDLNM_QP <- gam(y~te(TEMP,LAG,by=W,k=c(10,10)),family=quasipoisson,method="REML",data=tempDat) 

# Now grid of temperature-lag to predict over
temp_grid <- seq(-25,33,by=0.1)
lag_grid <- 0:max_lag
# Negative Binomial relative risk
NB_RR <- data.table(expand.grid(temp_grid,lag_grid))
names(NB_RR) <- c("TEMP","LAG")
NB_RR$W <- 1
n.sims <- 1000
set.seed(29) # to compare with quasiPoisson for the same seed
b <- rmvn(n.sims,coef(gamDLNM_NB),gamDLNM_NB$Vp)[,-1]
X <- predict(gamDLNM_NB,newdata=NB_RR,type="lpmatrix")[,-1]
LP <- X %*% t(b)
NB_RR$RR <- apply(exp(LP),1,mean) 
Lower <- apply(LP,1,quantile,probs=0.05)
Upper <- apply(LP,1,quantile,probs=0.95)
myDF <- data.frame(Upper,Lower)
NB_RR$sign <- as.numeric(with(myDF,Upper>=0 & Lower<=0)) # is zero in the 95% CrI?
NB_RR$model <- "Negative Binomial"
SignNB <- NB_RR[sign==0]
## Now quasi-Poisson
set.seed(29)
b <- rmvn(n.sims,coef(gamDLNM_QP),gamDLNM_QP$Vp)[,-1]
X <- predict(gamDLNM_QP,newdata=gam_grid,type="lpmatrix")[,-1]
LP <- X %*% t(b)
QP_RR <- copy(NB_RR)
QP_RR$RR <- apply(exp(LP),1,mean) 
Lower <- apply(LP,1,quantile,probs=0.05)
Upper <- apply(LP,1,quantile,probs=0.95)
myDF <- data.frame(Upper,Lower)
QP_RR$sign <- as.numeric(with(myDF,Upper>=0 & Lower<=0)) # is zero in the 95% CrI?
QP_RR$model <- "quasi-Poisson"
SignQP <- QP_RR[sign==0]
# Now plot
plotDat <- rbind(NB_RR,QP_RR)
Sign <- rbind(SignNB,SignQP)
myBrks <- seq(0.92,1.38,by=0.02)
EfSur <- ggplot(data=plotDat) + geom_raster(aes(x=TEMP, y = LAG, fill=RR)) + 
  scale_fill_gradient2(limits=range(myBrks),breaks=myBrks,midpoint=1,high="darkred",low="darkblue") + 
  xlab("Apparent Temperature (\u00B0C)") +  ylab("Lag (days)") + 
  geom_point(data=Sign,aes(x=TEMP,y=LAG),alpha=0.05) + 
  theme_bw() + facet_wrap(~model) +
  theme(axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold"),legend.text=element_text(size=12),
        legend.key.height = unit(2, 'cm'))
EfSur




library(mgcv)
library(dlnm)
library(data.table)
library(plot3D)
library(MASS)
library(tictoc)

# set this as desired
plotDir <- getwd()

### Simulate data such that the lagged effect of say temperature
### is spatially varying. Use existing data in package dlnm
### to get the baseline surface of temp and lag.
head(chicagoNMMAPS)
dat <- data.table(y=chicagoNMMAPS$cvd,temp0=chicagoNMMAPS$temp,pm10_0=chicagoNMMAPS$pm10)
n <- nrow(dat)
max_lag <- 15
for(i in 1:max_lag){
  dat[,paste("temp",i,sep="") := c(rep(NA,i),temp0[1:(n-i)]) ]
}
### Now fit the model 
index <- grep("temp", colnames(dat) )
tempDat <- na.omit( dat[,..index] )
yHere <- dat$y[-c(1:max_lag)]
LAG <- matrix(0:max_lag,nrow(tempDat),length(0:max_lag),byrow=TRUE) 
TEMP <- as.matrix( tempDat )
L <- matrix(1,length(0:max_lag),nrow=nrow(LAG),ncol=ncol(LAG)) 
model <- gam(yHere~te(TEMP,LAG,by=L,k=10),family=nb,method="REML") 
## 3D plot of RR
tempGrid <- seq(-27,34,by=0.2)
dlnm_grid <- data.table(expand.grid(tempGrid,0:max_lag))
names(dlnm_grid) <- c("TEMP","LAG")
dlnm_grid$L <- 1
## log-relative risk
Overall <- ( predict(model,newdata=dlnm_grid,type="lpmatrix")[,-1] %*% coef(model)[-1] )[,1]


## Make up some surfaces for 5 fictitious districts
# lists to store all the associated data sets
TEMPlist <- LAGlist <- Llist <- models <- list()
## district 1
## just use the data under different configurations. Start with subsetting the data 
index <- grep("temp", colnames(dat) )
tempDat <- na.omit( dat[,..index] )
yHere <- dat$y[-c(1:max_lag)]
toRemove <- c(1:500)
tempDatHere <- tempDat[-toRemove]
LAG <- matrix(0:max_lag,nrow(tempDatHere),length(0:max_lag),byrow=TRUE) 
TEMP <- as.matrix( tempDatHere )
L <- matrix(1,length(0:max_lag),nrow=nrow(LAG),ncol=ncol(LAG)) 
modelHere <- gam(yHere[-toRemove]~te(TEMP,LAG,by=L,k=10),family=nb,method="REML") 
models[[1]] <- modelHere
deviation1 <- ( predict(model,newdata=dlnm_grid,type="lpmatrix")[,-1] %*% coef(modelHere)[-1] )[,1]
## store the data
simDat <- tempDatHere
simDat$y <- yHere[-toRemove]
simDat[,district:= "district1"]
TEMPlist[[1]] <- TEMP
LAGlist[[1]] <- LAG
Llist[[1]] <- L
#####
##### Second, relabel some of the counts
datHere <- data.table(y=chicagoNMMAPS$cvd,temp0=chicagoNMMAPS$temp,pm10_0=chicagoNMMAPS$pm10)
n <- nrow(datHere)
datHere[,temp0 := -temp0 + 5]
max_lag <- 15
for(i in 1:max_lag){
  datHere[,paste("temp",i,sep="") := c(rep(NA,i),temp0[1:(n-i)]) ]
}
### Now fit the model 
index <- grep("temp", colnames(datHere) )
tempDat <- na.omit( datHere[,..index] )
yHere <- datHere$y[-c(1:max_lag)]
LAG <- matrix(0:max_lag,nrow(tempDat),length(0:max_lag),byrow=TRUE) 
TEMP <- as.matrix( tempDat )
L <- matrix(1,length(0:max_lag),nrow=nrow(LAG),ncol=ncol(LAG)) 
modelHere <- gam(yHere~te(TEMP,LAG,by=L,k=10),family=nb,method="REML") 
models[[2]] <- modelHere
deviation2 <- ( predict(modelHere,newdata=dlnm_grid,type="lpmatrix")[,-1] %*% coef(modelHere)[-1] )[,1]
## store the data
tempDat$y <- yHere
tempDat$district <- "district2"
simDat <- rbind(simDat,tempDat)
TEMPlist[[2]] <- TEMP
LAGlist[[2]] <- LAG
Llist[[2]] <- L
#####
##### Third, change counts for some of the temp ranges
datHere <- data.table(y=chicagoNMMAPS$cvd,temp0=chicagoNMMAPS$temp)
n <- nrow(datHere)
datHere[temp0>10 & temp0<20,y:=y+10]
max_lag <- 15
for(i in 1:max_lag){
  datHere[,paste("temp",i,sep="") := c(rep(NA,i),temp0[1:(n-i)]) ]
}
### Now fit the model 
index <- grep("temp", colnames(datHere) )
tempDat <- na.omit( datHere[,..index] )
yHere <- datHere$y[-c(1:max_lag)]
LAG <- matrix(0:max_lag,nrow(tempDat),length(0:max_lag),byrow=TRUE) 
TEMP <- as.matrix( tempDat )
L <- matrix(1,length(0:max_lag),nrow=nrow(LAG),ncol=ncol(LAG)) 
modelHere <- gam(yHere~te(TEMP,LAG,by=L,k=10),family=nb,method="REML") 
models[[3]] <- modelHere
deviation3 <- ( predict(modelHere,newdata=dlnm_grid,type="lpmatrix")[,-1] %*% coef(modelHere)[-1] )[,1]
## store the data
tempDat$y <- yHere
tempDat$district <- "district3"
simDat <- rbind(simDat,tempDat)
TEMPlist[[3]] <- TEMP
LAGlist[[3]] <- LAG
Llist[[3]] <- L
#####
##### Fourth, change counts for some of the temp ranges
datHere <- data.table(y=chicagoNMMAPS$cvd,temp0=chicagoNMMAPS$temp)
n <- nrow(datHere)
datHere[temp0>0 & temp0<10,y:=y+15]
max_lag <- 15
for(i in 1:max_lag){
  datHere[,paste("temp",i,sep="") := c(rep(NA,i),temp0[1:(n-i)]) ]
}
### Now fit the model 
index <- grep("temp", colnames(datHere) )
tempDat <- na.omit( datHere[,..index] )
yHere <- datHere$y[-c(1:max_lag)]
LAG <- matrix(0:max_lag,nrow(tempDat),length(0:max_lag),byrow=TRUE) 
TEMP <- as.matrix( tempDat )
L <- matrix(1,length(0:max_lag),nrow=nrow(LAG),ncol=ncol(LAG)) 
modelHere <- gam(yHere~te(TEMP,LAG,by=L,k=10),family=nb,method="REML") 
models[[4]] <- modelHere
deviation4 <- ( predict(modelHere,newdata=dlnm_grid,type="lpmatrix")[,-1] %*% coef(modelHere)[-1] )[,1]
## store the data
tempDat$y <- yHere
tempDat$district <- "district4"
simDat <- rbind(simDat,tempDat)
TEMPlist[[4]] <- TEMP
LAGlist[[4]] <- LAG
Llist[[4]] <- L
#####
##### Fifth, change counts for some of the temp ranges
datHere <- data.table(y=chicagoNMMAPS$cvd,temp0=chicagoNMMAPS$temp)
n <- nrow(datHere)
max_lag <- 15
for(i in 1:max_lag){
  datHere[,paste("temp",i,sep="") := c(rep(NA,i),temp0[1:(n-i)]) ]
}
datHere[temp15>0 & temp15<10 & temp14>0 & temp14<10&temp13>0 & temp13<10&
          temp12>0 & temp12<10,y:=y*2]
### Now fit the model 
index <- grep("temp", colnames(datHere) )
tempDat <- na.omit( datHere[,..index] )
yHere <- datHere$y[-c(1:max_lag)]
LAG <- matrix(0:max_lag,nrow(tempDat),length(0:max_lag),byrow=TRUE) 
TEMP <- as.matrix( tempDat )
L <- matrix(1,length(0:max_lag),nrow=nrow(LAG),ncol=ncol(LAG)) 
modelHere <- gam(yHere~te(TEMP,LAG,by=L,k=10),family=nb,method="REML") 
models[[5]] <- modelHere
deviation5 <- ( predict(modelHere,newdata=dlnm_grid,type="lpmatrix")[,-1] %*% coef(modelHere)[-1] )[,1]
## store the data
tempDat$y <- yHere
tempDat$district <- "district5"
simDat <- rbind(simDat,tempDat)
TEMPlist[[5]] <- TEMP
LAGlist[[5]] <- LAG
Llist[[5]] <- L



#### Plot the 3D surfaces
# manual colours
myBrks <- seq(0.86,1.5,by=0.01)
myColsFun <- colorRampPalette(c("darkblue","blue","white","red", "darkred"))
maxDev <- max( c(max(myBrks)-1,1-min(myBrks)) )
allBrks <- seq(1-maxDev,1+maxDev,by=0.01  )
Cols <- myColsFun(length(allBrks)-1)
ColsSub <- Cols[allBrks[-1] > min(myBrks)]
# Original model
x11(width=16,height=8)
par(mar=c(1,2.8,0,0),mfrow=c(2,4))
persp3D(tempGrid, 0:max_lag, matrix(exp(Overall), length(tempGrid), length(0:max_lag)), 
        theta = 200, phi = 20,ticktype = "detailed", xlab = "temp", ylab = "lag", colkey = F,cex.axis = 1.5,cex.lab=1.5,
        zlab = "RR", expand = 2/3, shade = 0.1, zlim=c(0.9,1.5),main="",col=ColsSub,breaks = myBrks)
text3D(x=95,y=0,z=1.42,"a) chicagoNMMAPS",add=T,cex=1.5)
# Simulated surfaces
districtLogRR <- list(deviation1,deviation2,deviation3,deviation4,deviation5)
labels <- c("b)","c)","d)","e)","f)")
for(i in 1:5){
persp3D(tempGrid, 0:max_lag, matrix(exp(districtLogRR[[i]]), length(tempGrid), length(0:max_lag)), 
        theta = 200, phi = 20,ticktype = "detailed", xlab = "temp", ylab = "lag", colkey = F,cex.axis = 1.5,cex.lab=1.5,
        zlab = "RR", expand = 2/3, shade = 0.1, zlim=c(0.9,1.5),main="",breaks = myBrks,col=ColsSub)
text3D(x=95,y=0,z=1.42,paste(labels[i]," district ",i,sep=""),add=T,cex=1.5)
}
## and the bar
colkey(at=myBrks,add=F,side=4,clim=range(myBrks),col=ColsSub,length=0.95,width=5,cex.axis = 1.4)



## Now simulate some counts
districts <- unique(simDat$district)
for(i in 1:5){
  pred <- predict(models[[i]],newdata=list(TEMP=TEMPlist[[i]],LAG=LAGlist[[i]],L=Llist[[i]]),type="response")
  set.seed(29)
  n <- nrow(simDat[district==districts[i],])
  simDat[district==districts[i],y:=rpois(n,pred)]
}


# ## Fit an overall model
# # sort out the covariates
# TEMP <- do.call(rbind,TEMPlist) 
# L <- do.call(rbind,Llist)
# LAG <- do.call(rbind,LAGlist)
# # Overall model
# Omodel <- gam(y~te(TEMP,LAG,by=L,k=c(10),bs=c("tp","tp"),m=2),
#              family=nb,data=simDat,method="REML")  
# ov_grid <- data.table(expand.grid(tempGrid,0:max_lag))
# names(ov_grid) <- c("TEMP","LAG")
# ov_grid[,L:=1]
# modelMatrix <- predict(Omodel,newdata=ov_grid,type="lpmatrix")
# ov_eff_index <- grep("te(TEMP,LAG)",colnames(modelMatrix),fixed=T)
# overallEffects <- exp(  modelMatrix[,ov_eff_index] %*% (coef(Omodel))[ov_eff_index] ) 
# x11(width=8,height=6.6)
# par(mar=c(1,2.8,0,0))
# persp3D(tempGrid, 0:max_lag, matrix(overallEffects, length(tempGrid), length(0:max_lag)), 
#         theta = 200, phi = 20,ticktype = "detailed", xlab = "temp", ylab = "lag", colkey = F,cex.axis = 1.5,cex.lab=1.5,
#         zlab = "RR", expand = 2/3, shade = 0.1, zlim=c(0.9,1.5),main="",col=ColsSub,breaks = myBrks)
# text3D(x=95,y=0,z=1.42,"h) global model",add=T,cex=1.5)
# dev.print(pdf,file=file.path(plotDir,"fig6h.pdf"))


###############################################################
### Fit a hierarchical model using tensor products

# We need to create a district matrix that is a factor.
simDat[,district:=factor(district)]
DISTRICT <- rep(simDat$district,max_lag+1)
dim(DISTRICT) <- c(nrow(simDat),max_lag+1)
## Now all the other covariates
TEMP <- do.call(rbind,TEMPlist)
L <- do.call(rbind,Llist)
LAG <- do.call(rbind,LAGlist)
n_knots <- 7
system.time(
  pooling_model <- gam(y~s(district,bs="re")+te(TEMP,LAG,by=L,k=n_knots,bs=c("tp","tp"),m=2)+
                  te(TEMP,LAG,DISTRICT,by=L,bs=c("tp","tp","re"),k=c(n_knots,n_knots,5)),
                  family=poisson,data=simDat,method="REML") 
)
k.check(pooling_model)
# Now the estimates
covariate_grid <- data.table(expand.grid(tempGrid,0:max_lag,levels(simDat$district)))
names(covariate_grid) <- c("TEMP","LAG","DISTRICT")
covariate_grid[,DISTRICT:=factor(DISTRICT)]
covariate_grid$L <- 1
covariate_grid[,district := DISTRICT]
# compute the relative risk without any intercepts
whichInterepts <- c(1,grep("s(district)", names(pooling_model$coefficients),fixed=T))
modelMatrix <- predict(pooling_model,newdata=covariate_grid,type="lpmatrix")[,-whichInterepts] 
covariate_grid$RR <- exp(  modelMatrix %*% coef(pooling_model)[-whichInterepts] )
# and the global relative risk
ov_grid <- copy( covariate_grid[DISTRICT=="district1"] ) ## doesn't matter which district
modelMatrix <- predict(pooling_model,newdata=ov_grid,type="lpmatrix")
ov_eff_index <- grep("te(TEMP,LAG)",colnames(modelMatrix),fixed=T)
globalRR <- exp(  modelMatrix[,ov_eff_index] %*% (coef(pooling_model))[ov_eff_index] ) 


#########################################################
#### Plot the results
# range of RR estimates to create the breaks for the plot
range( covariate_grid$RR )
x11(width=16,height=8)
par(mar=c(1,2.8,0,0),mfrow=c(2,4))
# global term
persp3D(tempGrid, 0:max_lag, matrix(globalRR, length(tempGrid), length(0:max_lag)), 
        theta = 200, phi = 20,ticktype = "detailed", xlab = "temp", ylab = "lag", colkey = F,cex.axis = 1.5,cex.lab=1.5,
        zlab = "RR", expand = 2/3, shade = 0.1, zlim=c(0.9,1.5),main="",col=ColsSub,breaks = myBrks)
text3D(x=95,y=0,z=1.42,"g) global estimate",add=T,cex=1.5)
# global plus deviations
persp3D(tempGrid, 0:max_lag, matrix(covariate_grid[DISTRICT=="district1",RR], length(tempGrid), length(0:max_lag)), 
        theta = 200, phi = 20,ticktype = "detailed", xlab = "temp", ylab = "lag", colkey = F,cex.axis = 1.5,cex.lab=1.5,
        zlab = "RR", expand = 2/3, shade = 0.1, zlim=c(0.9,1.5),main="",col=ColsSub,breaks = myBrks)
text3D(x=95,y=0,z=1.42,"h) district 1",add=T,cex=1.5)
persp3D(tempGrid, 0:max_lag, matrix(covariate_grid[DISTRICT=="district2",RR], length(tempGrid), length(0:max_lag)), 
        theta = 200, phi = 20,ticktype = "detailed", xlab = "temp", ylab = "lag", colkey = F,cex.axis = 1.5,cex.lab=1.5,
        zlab = "RR", expand = 2/3, shade = 0.1, zlim=c(0.9,1.5),main="",col=ColsSub,breaks = myBrks)
text3D(x=95,y=0,z=1.42,"i) district 2",add=T,cex=1.5)
persp3D(tempGrid, 0:max_lag, matrix(covariate_grid[DISTRICT=="district3",RR], length(tempGrid), length(0:max_lag)), 
        theta = 200, phi = 20,ticktype = "detailed", xlab = "temp", ylab = "lag", colkey = F,cex.axis = 1.5,cex.lab=1.5,
        zlab = "RR", expand = 2/3, shade = 0.1, zlim=c(0.9,1.5),main="",col=ColsSub,breaks = myBrks)
text3D(x=95,y=0,z=1.42,"j) district 3",add=T,cex=1.5)
persp3D(tempGrid, 0:max_lag, matrix(covariate_grid[DISTRICT=="district4",RR], length(tempGrid), length(0:max_lag)), 
        theta = 200, phi = 20,ticktype = "detailed", xlab = "temp", ylab = "lag", colkey = F,cex.axis = 1.5,cex.lab=1.5,
        zlab = "RR", expand = 2/3, shade = 0.1, zlim=c(0.9,1.5),main="",col=ColsSub,breaks = myBrks)
text3D(x=95,y=0,z=1.42,"k) district 4",add=T,cex=1.5)
persp3D(tempGrid, 0:max_lag, matrix(covariate_grid[DISTRICT=="district5",RR], length(tempGrid), length(0:max_lag)), 
        theta = 200, phi = 20,ticktype = "detailed", xlab = "temp", ylab = "lag", colkey = F,cex.axis = 1.5,cex.lab=1.5,
        zlab = "RR", expand = 2/3, shade = 0.1, zlim=c(0.9,1.5),main="",col=ColsSub,breaks = myBrks)
text3D(x=95,y=0,z=1.42,"l) district 5",add=T,cex=1.5)





library(mgcv)
library(dlnm)
library(data.table)
library(plot3D)

### Simulate data such that the lagged effect of temperature
### is spatially varying in a smooth way. Use the Chicago data 
### from package dlnm to get the baseline surface of temp and lag.

# First put the data in a clean data.table
head(chicagoNMMAPS)
dat <- data.table(y=chicagoNMMAPS$cvd,temp0=chicagoNMMAPS$temp)
n <- nrow(dat)
max_lag <- 15
for(i in 1:max_lag){
  dat[,paste("temp",i,sep="") := c(rep(NA,i),temp0[1:(n-i)]) ]
}
### Now fit a DLNM GAM to get the estimated temp-lag surface
index <- grep("temp", colnames(dat) )
tempDat <- na.omit( dat[,..index] )
yHere <- dat$y[-c(1:max_lag)]
LAG <- matrix(0:max_lag,nrow(tempDat),length(0:max_lag),byrow=TRUE) 
TEMP <- as.matrix( tempDat )
L <- matrix(1,length(0:max_lag),nrow=nrow(LAG),ncol=ncol(LAG)) 
model <- gam(yHere~te(TEMP,LAG,by=L,k=10),family=nb) 
## Plot the relative risk
temp_grid <- seq(-27,40,by=0.2)
lag_grid <- 0:max_lag
dlnm_grid <- data.table(expand.grid(temp_grid,lag_grid))
names(dlnm_grid) <- c("TEMP","LAG")
dlnm_grid$L <- 1
logRR <- ( predict(model,newdata=dlnm_grid,type="lpmatrix")[,-1] %*% coef(model)[-1] )[,1]
dlnm_grid$RR <- exp(logRR)
persp3D(temp_grid, lag_grid, matrix(dlnm_grid$RR, length(temp_grid), length(lag_grid)), 
        theta = 200, phi = 20,ticktype = "detailed", xlab = "temp", ylab = "lag", 
        zlab = "RR", expand = 2/3, shade = 0.5)


## A smooth function over some coordinates
## This will be used to induce similarity in the estimated coefficients 
## from the fitted model above, and therefore create smoothly varying
## relative risk surfaces
SS <- function(x,z,sx=0.3,sz=0.4) { 
  (pi**sx*sz)*(1.2*exp(-(x-0.2)^2/sx^2-(z-0.3)^2/sz^2)+
                 0.8*exp(-(x-0.7)^2/sx^2-(z-0.8)^2/sz^2))
}
gridResol <- 5
## Fake coordinates for 25 grid cells
x <- seq(0,1,len=gridResol)
z <- seq(0,1,len=gridResol)
Grid <- data.table(expand.grid(x,z))
names(Grid) <- c("x","z")
spatialS <- SS(Grid$x,Grid$z)
# center it
spatialS <- spatialS-mean(spatialS)
spatialS_matrix <- matrix(spatialS,gridResol,gridResol)
# and plot
persp3D(x, z, (spatialS_matrix/sd(spatialS_matrix))*2*coef(model)[2]+coef(model)[2], 
        theta = 200, phi = 20,ticktype = "detailed", xlab = "x", ylab = "z", 
        zlab = "coef", expand = 2/3, shade = 0.5)
## now use this surface to "jitter" the coefficients of the estimated DLNM-GAM 
## to create a list of different-yet-similar coefficients for each grid cell
coef_list <- list()
nCells <- nrow(Grid)
for(i in 1:nCells){
  coef_list[[i]] <- (spatialS[i]/sd(spatialS))*0.5*coef(model)+coef(model)
}
# plot the resulting relative risk surfaces for each grid cell 
x11(width=15,height=15); par(mfrow=c(5,5))
X <- predict(model,newdata=dlnm_grid,type="lpmatrix")[,-1]
myBrks <- seq(0.80,3.7,by=0.01)
for(i in 1:25){
  RR <- exp( ( X  %*% coef_list[[i]][-1] )[,1] )
  persp3D(temp_grid, lag_grid, matrix(RR, length(temp_grid), length(lag_grid)), 
        theta = 200, phi = 20,ticktype = "detailed", xlab = "Temp", ylab = "Lag", 
        zlab = "effect", expand = 2/3, shade = 0.5,zlim=range(myBrks),breaks = myBrks,colkey = F)
}

## Now simulate some counts using these relative risk surfaces fro a Poisson 
myDat <- data.table(chicagoNMMAPS)
## Thin the data as it's a bit too large
myDat <- myDat[year<1990]
## First, a temperature GAM to simulate temperature values in the different 
## grid cells. Each cell will have its own temperature time series with the 
## same statistical properties.
tempModDat <- data.table(temp=myDat$temp,time=seq(0,1,len=nrow(myDat)))
tempMod <- gam(temp ~ s(time,k=200),data=tempModDat)
## plot the original temp time series against a simulation from the 
## temperature GAM to understand what this does.
plot( simulate(tempMod)[,1],tempModDat$temp,pch=20 )
## Now simulate the counts
data_list <- list()
n <- nrow(myDat)
for(i in 1:nCells){
  if(i%%10==0){print(paste(i," out of ",nCells,sep=""))}
  data_list[[i]] <- data.table(y=-99.9,temp0=simulate(tempMod)[,1])
  for(j in 1:max_lag){
    data_list[[i]][,paste("temp",j,sep="") := c(rep(NA,j),temp0[1:(n-j)]) ]
  }
  index <- grep("temp", colnames(data_list[[i]]) )
  tempDat <- na.omit( data_list[[i]][,..index] )
  LAG <- matrix(0:max_lag,nrow(tempDat),length(0:max_lag),byrow=TRUE) 
  TEMP <- as.matrix( tempDat )
  L <- matrix(1,length(0:max_lag),nrow=nrow(LAG),ncol=ncol(LAG)) 
  LPmatrix <- predict(model,newdata=list(TEMP=TEMP,LAG=LAG,L=L),type="lpmatrix")
  meanHere <- exp( (LPmatrix%*%coef_list[[i]])[,1] )
  yHere <- rpois(length(meanHere),lambda=meanHere)
  data_list[[i]]$y <- c(rep(NA,max_lag),yHere)
}

### Now see if a DLNM-GAM with a lon-lat interaction can pick up the spatially 
### varying relationship

## First, create the coordinate covariates
lons <- seq(0,1,len=gridResol)
lats <- seq(0,1,len=gridResol)
lonlat <- data.table(expand.grid(lons,lats))
names(lonlat) <- c("lon","lat")
for(i in 1:nCells){
  data_list[[i]]$lon <- lonlat$lon[i]
  data_list[[i]]$lat <- lonlat$lat[i]
}
## now put all the data together in single data.table
allDat <- do.call(rbind,data_list)
## give each cell a unique ID
lonlat[,id:= 1:nCells]
allDat <- merge(allDat,lonlat,by=c("lon","lat"))
## Now fit the model. We will iclude a spatially varying smooth term (intercept)
## plus an interacion between space, temperature and lag.
index <- grep("temp", colnames(allDat) )
tempDat <- na.omit( allDat[,..index] )
yHere <- na.omit(allDat$y)
LAG <- matrix(0:max_lag,nrow(tempDat),length(0:max_lag),byrow=TRUE) 
TEMP <- as.matrix( tempDat )
L <- matrix(1,length(0:max_lag),nrow=nrow(LAG),ncol=ncol(LAG)) 
LON <- matrix(na.omit(allDat)$lon,nrow=nrow(L),ncol=ncol(LAG))
LAT <- matrix(na.omit(allDat)$lat,nrow=nrow(L),ncol=ncol(LAG))
lon <- na.omit(allDat)$lon
lat <- na.omit(allDat)$lat
## The model is heavy so fit with bam and use 12 threads
system.time(
  modelS <- bam(yHere~s(lon,lat,k=20)+te(TEMP,LAG,LON,LAT,by=L,bs=c("tp","tp","tp"),d=c(1,1,2),k=c(10,10,10)),
                family=poisson,method="fREML",discrete=T,nthreads = 12)
)
## Plot the estimated relative risk
## create an index to pick the terms without the intercept or the "purely
## spatial" term, so that the relative risk and not the absolute risk 
## is computed.
spatialIndex <- grep("te(TEMP,LAG,LON,LAT)",names(coef(modelS)),fixed=T)
b <- coef(modelS)[spatialIndex]
x11(width=15,height=15); par(mfrow=c(5,5))
for(i in 1:25){
  dlnm_grid$LON <- lonlat[id==i,"lon"]
  dlnm_grid$LAT <- lonlat[id==i,"lat"]
  dlnm_grid$lon <- lonlat[id==i,"lon"]
  dlnm_grid$lat <- lonlat[id==i,"lat"]
  X <- predict(modelS,newdata=dlnm_grid,type="lpmatrix")[,spatialIndex]
  RR <- exp( ( X %*% b )) [,1]
  persp3D(temp_grid, lag_grid, matrix(RR, length(temp_grid), length(lag_grid)), 
          theta = 200, phi = 20,ticktype = "detailed", xlab = "Temp", ylab = "Lag", 
          zlab = "effect", expand = 2/3, shade = 0.5,zlim=range(myBrks),breaks = myBrks,colkey = F)
}

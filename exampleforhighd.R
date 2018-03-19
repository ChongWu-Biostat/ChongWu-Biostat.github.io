########################################################
# Statistical Learning and Data Mining Codes        ####
# Author: Xiaotong Shen, Chen Gao, Chong Wu         ####
# We illustrate how to use regularization in R      ####
########################################################


########################################################
# p30 Lasso
########################################################

#install.packages("glmnet")
library(glmnet) #use coordinate descent
data(BinomialExample)

dim(x)

dim(y)
# Generation response Y and Design matrix X, Y= beta
fit.lasso=glmnet(x,y)
coef(fit.lasso,s=0.01)
coef(fit.lasso,s=0.1)
predict(fit.lasso,newx=x[1:10,],s=c(0.01,0.1))


fit.lasso = glmnet(x,y, family="binomial")

plot(fit.lasso, xvar ="norm")
plot(fit.lasso, xvar ="lambda")
plot(fit.lasso, xvar = "dev")

# cross validation

# Note also that the results of cv.glmnet are random,
# since the folds are selected at random.
# Users can reduce this randomness by running cv.glmnet many times,
# and averaging the error curves.

cvfit = cv.glmnet(x, y)
plot(cvfit)
cvfit$lambda.min

coef(cvfit, s = "lambda.min")


# Reference: https://web.stanford.edu/~hastie/glmnet

########################################################
# p40 lasso
########################################################
library("lars") #close relates to homotopy method; Least Angle Regression algorithm
#data(diabetes)
#attach(diabetes)

#y quantitative measure of disease progression one year after baseline
# construct a model that predicted response y from covariates x1, x2, . . . , x10
library(glmnet)
lasso.fit<-glmnet(x, y, family="gaussian")
coef(lasso.fit, s=0.01)
predict(lasso.fit,newx=x,s=c(0.01,0.1))

# default
plot(lasso.fit, xvar="norm" ) 

# plot: coef’s vs log lambda
plot(lasso.fit, xvar="lambda")

plot(lasso.fit, xvar="dev")



tb= Sys.time()
cv.lasso.fit<-cv.glmnet(x, y, nfold=10,family="gaussian")
Sys.time()-tb

yhat.lasso<-predict(cv.lasso.fit, newx=x)

########################################################
# p42 lars
########################################################

library("lars") #Efficient procedures for fitting an entire lasso sequence with the cost of a single least squares fit
# Trevor Hastie

lars.fit <- lars(x,y, type="lasso")

# Note also that the results of cvlars are random

tb=Sys.time()
cv.lars.fit<-cv.lars(x, y, K=10,index=seq(from=0, to=1, length=80))
Sys.time()-tb


# choose fraction based min cv error rule
min.indx <- which.min(cv.lars.fit$cv)
s.cvmin <- cv.lars.fit$index[min.indx]

yhat.lars<-predict(lars.fit, newx=x, s=s.cvmin,type="fit", mode="fraction")

# choose fraction based 1-se cv error rule
# largest value of lambda such that
# error is within 1 standard error of the minimum:

cv1se <- min(cv.lars.fit$cv) + cv.lars.fit$cv.error[min.indx]
indx2 <- cv.lars.fit$cv<cv1se
s.cvmin <- max(cv.lars.fit$index[indx2])
yhat.lars<-predict(lars.fit, newx=x, s=s.cvmin,
                   type="fit", mode="fraction")

plot(y, yhat.lars$fit, col = "purple", pch =16)

# My opinionn: glmnet is more powerful and popular than lars

########################################################
# p44 grpreg
########################################################

rm(list = ls())

library(grpreg)
library("lars")
data(diabetes)
attach(diabetes)

# x1-x4: age, sex, BMI, BP;
# x5-x10: serum measurements

group <- c(rep(1,4), rep(2,6))
par(mfrow=c(2,3))

fit <- grpreg(x,y,group,penalty="grLasso") #will have some problems

x = as.data.frame(x)
x = as.matrix(x)
fit <- grpreg(x,y,group,penalty="grLasso")

plot(fit,main = "Group Lasso")

fit <- grpreg(x,y,group,penalty="grMCP") # The former involves an MCP penalty being applied to an L2-norm of each group.
plot(fit, main = "Group MCP")

fit <- grpreg(x,y,group,penalty="grSCAD")
plot(fit, main = "Group SCAD")

# bi-level selection
# Group exponential lasso

#Bi-level means carrying out variable selection at the group level as well as the level of individual covariates (i.e., selecting important groups as well as important members of those groups)
#Group selection selects important groups, and not members within the group – i.e., within a group, coefficients will either all be zero or all nonzero.

fit <- grpreg(x,y,group,penalty="gel") #Group exponential lasso
plot(fit, main ="gel")

fit <- grpreg(x,y,group,penalty="cMCP") # a hierarchical penalty which places an outer MCP penalty on a sum of inner MCP penalties for each group
plot(fit, main ="cMCP")

dev.off()

#However, especially when p is large compared with n, grpreg may fail to converge at low values of lambda, where models are nonidentifiable or nearly singular. Often, this is not the region of the coefficient path that is most interesting.

res <- select(fit, criterion = "AIC")
res$lambda

# cross-validation
# default penalty is grLasso

par(mfrow=c(1,2))

cvfit <- cv.grpreg(x, y, group, seed =12345)

# lambda based on minimum cv error rule
cvfit$lambda.min
cvfit$cve
cvfit$cvse

plot(cvfit)
summary(cvfit)
coef(cvfit) ## Beta at minimum CVE

cvfit <- cv.grpreg(x, y, group, penalty = "grSCAD")
plot(cvfit)
summary(cvfit)
coef(cvfit) ## Beta at minimum CVE


dev.off()

# reference :https://cran.r-project.org/web/packages/grpreg/vignettes/quick-start.pdf
# Penalty : https://cran.r-project.org/web/packages/grpreg/vignettes/penalties.pdf

########################################################
# p49 SGL
########################################################

library(SGL) # sparse group lasso, similar as elastic net.

data <- list(x=x, y=y)
index <- c(rep(1,4), rep(2,6))
fit <- SGL(data, index, type = "linear")

# somehow much slower than cv.grpreg
cvFit <- cvSGL(data, index, type = "linear",nfold = 10)

########################################################
# p50 fused lasso
########################################################

library("genlasso")
set.seed(1)

n = 100

i = 1:n

y1 = (i > 20 & i < 30) + 5*(i > 50 & i < 70) + rnorm(n, sd=0.1)

out = fusedlasso1d(y1)
# In the common signal approximator case, X = I, we assume that the observed data y = (y1, . . . yn) ∈ Rn is generated from a process whose mean changes at only a smaller number of locations, when ordered sequentially from 1 to n.

plot(out, lambda=1, col ="red", pch =16)


set.seed(1)
y1 = matrix(runif(256), 16, 16)
i = (row(y1) - 8.5)^2 + (col(y1) - 8.5)^2 <= 4^2
y1[i] = y1[i] + 1

out = fusedlasso2d(y1)

co = coef(out, nlam=5)

par(mfrow=c(2,3))
image(y1, col = terrain.colors(12), main = "y1")
for(i in 1:5){
  image(matrix(co$beta[,i],nrow=16),
        col = terrain.colors(12),
        main = paste("lambda = ", round(co$lambda[i], 6)))
}

# reference: https://cran.r-project.org/web/packages/genlasso/vignettes/article.pdf


########################################################
# p54 elastic net
########################################################

library(elasticnet)
data(diabetes)
attach(diabetes)

cv.enet.fit<-cv.enet(x,y,lambda=0.05,
                     s=seq(0,1,length=100),mode="fraction",
                     trace=TRUE,max.steps=80)
cv.enet.fit$cv
cv.enet.fit$cv.error


library(glmnet)
cv.lasso.fit<-cv.glmnet(x, y, alpha =0.5, nfold=10,family="gaussian")





########################################################
# p55 SCAD, MCP
########################################################

library("lars")
data(diabetes)
attach(diabetes)
class(x) = "matrix"

library(ncvreg)

# default is penalty = "MCP", gamma = 3 for MCP
par(mfrow=c(1,2))
fit <- ncvreg(x,y, family="gaussian", gamma=3)
plot(fit, main = "MCP, gamm = 3")
fit <- ncvreg(x,y,family="gaussian",penalty="SCAD",gamma =3.7)
plot(fit, main ="SCAD, gamma = 3.7")


cv.ncvreg.fit<-cv.ncvreg(x, y, family="gaussian",
                         nfolds=10, seed=1, returnY=FALSE,
                         trace=FALSE)

cv.ncvreg.fit$cve
cv.ncvreg.fit$cvse

# choose lambda based min cv error rule

cv.ncvreg.fit$lambda.min


########################################################
# p92 FGSG
########################################################

library(FGSG) #Implement algorithms for  feature grouping and selection over an undirected graph
library(MASS)

X<-matrix(rnorm(25),5,5) # Design matrix

y<-rnorm(5) # response

tp<-c(1,2,2,3,3,4,4,5) # Specify graph

ncTFGS(X,y,tp,0.3,0.5)

#reference: https://cran.r-project.org/web/packages/genlasso/vignettes/article.pdf

#######################################################
# Truncated Lasso
#######################################################
library(glmtlp)
data("QuickStartExample")
fit = glmTLP(x,y)
plot(fit)
cvfit = cv.glmTLP(x, y,tau = 1)
plot(cvfit)

# documentation http://www.tc.umn.edu/~wuxx0845/glmtlp


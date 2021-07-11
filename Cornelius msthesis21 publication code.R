library(ISLR)
library(tree)
library(randomForest)
library(gbm)
library(glmnet)
library(readxl)
library(MASS)
library(class)
library(adabag)
library(e1071)
library(XLConnect)
library(pROC)
library(cluster)
library(tidyverse)
library(cluster)
library(writexl)
library(dplyr)
library(data.table)
library(neuralnet)
library(factoextra)
set.seed(9)


#reading the data
data <- read.csv(file="combined_CDC_data",fileEncoding = "UTF-8-BOM",na.strings = c('NA','','Missing'))
#eliminate non-states
data<-data[-which(data$res_state==c("MP","GU","PR","VI")),]
#changing dates to numeric
data[,3]<-as.numeric(data[,3])
data[,4]<-as.numeric(data[,4])
data[,6]<-as.numeric(data[,6])
data[,7]<-as.numeric(data[,7])

#trim data function to eliminate observations with missing values
trim.data<-function(data.in,col.select){
  fn.data<-data.in[,col.select]
  fn.data<-na.omit(fn.data)
  for (i in 1:ncol(fn.data)){
    if (class(fn.data[,i])=="factor"){
      fn.data<-fn.data[which(fn.data[,i]!="Unknown"),]
    }
  }
  fn.data<-droplevels(fn.data)
  return(fn.data)
}
#get trimmed data selecting which features we want
data.working<-trim.data(data,c(1,3:6,8:10,28,31,32))

#get smaller sample of trimmed data
data.sample<-data.working[sample(1:nrow(data.working),10000,replace=FALSE),]

#function to get clusters and attach as column to data
cluster.data<-function(data.in,n.clusters){
  rf.unsup<-randomForest(x=data.in[,-c(7:8)],mtry = ncol(data.in)-2,ntree = 500,proximity = TRUE)
  rf.prox<-rf.unsup$proximity
  clara.rf<-clara(rf.prox,n.clusters,cluster.only = FALSE, samples = 5)
  data.in.cluster<-cbind(data.in,clara.rf$clustering)
  print(summary(silhouette(clara.rf)))
  return(data.in.cluster)
}

#tune for number of clusters by looking at printed silhouette width
for (i in 2:8){
  cluster.data(data.sample,i)
}

#get clustered data settling on 4 clusters
data.clustered<-cluster.data(data.sample,4)


#build the RF classifier for each cluster \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

#get train and test vectors
train<-sort(sample(nrow(data.clustered),nrow(data.clustered)/2))
test<-rep(NA,nrow(data.clustered))
for (i in 1:length(test)){
  if (is.element(i,train)==FALSE){
    test[i]<-i
  } else {
    test[i]<-0
  }
}
test<-test[test!=0]


#baseline RF model with adaboost (what Iwendi did)
data.onlyDeath<-data.clustered[,-c(7,12)]
data.onlyDeath$death<-1
for (i in 1:nrow(data.onlyDeath)){
  if (data.onlyDeath$death_yn[i]=='No'){
    data.onlyDeath$death[i]<-0
  }
}

data.onlyDeath$death<-as.factor(data.onlyDeath$death)
rf.adaboost<-boosting(death~.,data = data.onlyDeath[train,-7],mfinal = 100)
rf.adaboost.predict<-predict(rf.adaboost,newdata = data.onlyDeath[test,-7])
table(rf.adaboost.predict$class,data.onlyDeath[test,11])

#cluster 1 RF model
intx.train<-intersect(train,which(data.clustered[,12]==1))
intx.test<-intersect(test,which(data.clustered[,12]==1))

data.onlyDeath<-data.clustered[,-c(7,12)]
full.rf<-randomForest(data.onlyDeath$death_yn~.,data=data.onlyDeath,subset = intx.train,ntree=1000,mtry=3)
full.rf.pred<-predict(full.rf,newdata = data.onlyDeath[intx.test,])
table(full.rf.pred,data.onlyDeath[intx.test,7])
# tp 45 tn 4090 fp 45 fn 155

#cluster 2 RF model
intx.train<-intersect(train,which(data.clustered[,12]==2))
intx.test<-intersect(test,which(data.clustered[,12]==2))

data.onlyDeath<-data.clustered[,-c(7,12)]
full.rf<-randomForest(data.onlyDeath$death_yn~.,data=data.onlyDeath,subset = intx.train,ntree=1000,mtry=3)
full.rf.pred<-predict(full.rf,newdata = data.onlyDeath[intx.test,])
table(full.rf.pred,data.onlyDeath[intx.test,7])
# tp 0 tn 512 fp 0 fn 0
summary(data.clustered[data.clustered[,12]==2,8])

#cluster 3 RF model
intx.train<-intersect(train,which(data.clustered[,12]==3))
intx.test<-intersect(test,which(data.clustered[,12]==3))

data.onlyDeath<-data.clustered[,-c(7,12)]
full.rf<-randomForest(data.onlyDeath$death_yn~.,data=data.onlyDeath,subset = intx.train,ntree=1000,mtry=3)
full.rf.pred<-predict(full.rf,newdata = data.onlyDeath[intx.test,])
table(full.rf.pred,data.onlyDeath[intx.test,7])
# tp 62 tn 60 fp 23 fn 23

#cluster 4 RF model
intx.train<-intersect(train,which(data.clustered[,12]==4))
intx.test<-intersect(test,which(data.clustered[,12]==4))

data.onlyDeath<-data.clustered[,-c(7,12)]
full.rf<-randomForest(data.onlyDeath$death_yn~.,data=data.onlyDeath,subset = intx.train,ntree=1000,mtry=3)
full.rf.pred<-predict(full.rf,newdata = data.onlyDeath[intx.test,])
table(full.rf.pred,data.onlyDeath[intx.test,7])
summary(data.clustered[data.clustered[,12]==4,8])
# tp 0 tn 442 fp 0 fn 0

#overall clustering RF results
tp<-45+62
tn<-4090+512+60+442
fp<-45+23
fn<-155+23
acc<-(tp+tn)/(tp+tn+fp+fn)  #0.95
prec<-tp/(tp+fp)            #0.61
rec<-tp/(tp+fn)             #0.38
f1<-2*(prec*rec)/(prec+rec) #0.47
table(acc,prec,rec,f1)


#get cluster variable mean
getcluster.var.mean<-function(data.in,n.clusters,var.num){
  storage.list<-list()
  if(class(data.in[,var.num])=="factor"){
    cnt<-nlevels(data.in[,var.num])
    for(i in 1:cnt){
      storage.vec<-rep(NA,n.clusters)
      for(j in 1:n.clusters){
        storage.vec[j]<-nrow(data.in[(data.in[,ncol(data.in)]==j) & (data.in[,var.num]==levels(data.in[,var.num])[i]),])/nrow(data.in[data.in[,ncol(data.in)]==j,])
      }
      storage.list[[i]]<-storage.vec
    }
    out.df<-as.data.frame(storage.list,col.names = levels(data.in[,var.num]))
  } else {
    storage.vec<-rep(NA,n.clusters)
    for (k in 1:n.clusters){
      storage.vec[k]<-mean(data.in[data.in[,ncol(data.in)]==k,var.num])
    }
    storage.list[[1]]<-storage.vec
    out.df<-as.data.frame(storage.list,col.names = colnames(data.in)[var.num])
  }
  return(out.df)
}

#get data frame with all variable means
get.means.df<-function(data.in,n.vars,n.clusters){
  list.store<-list()
  for(i in 1:(n.vars)){
    if(i!=10){
      list.store[[i]]<-getcluster.var.mean(data.in,n.clusters,i)
    }
  }
  return(bind_cols(list.store))
}
data.cluster.df<-get.means.df(data.clustered,11,5)

#get data frame with all variable means
get.means.df<-function(data.in,n.vars,n.clusters){
  list.store<-list()
  for(i in 1:(n.vars)){
    if(i!=10){
      list.store[[i]]<-getcluster.var.mean(data.in,n.clusters,i)
    }
  }
  return(bind_cols(list.store))
}
data.cluster.df<-get.means.df(data.clustered,11,5)

#get main sample of 10,000 means
get.sample.means<-function(data.in,n.vars){
  storage.list<-list()
  for(i in 1:n.vars){
    if(class(data.in[,i])=="factor"){
      cnt<-nlevels(data.in[,i])
      storage.vec<-rep(NA,cnt)
      for(j in 1:cnt){
        storage.vec[j]<-nrow(data.in[data.in[,i]==levels(data.in[,i])[j],])/nrow(data.in)
      }
      storage.list[[i]]<-transpose(as.data.frame(storage.vec,col.names = levels(data.in[,i])))
    } else {
      storage.num<-mean(data.in[,i])
      storage.list[[i]]<-transpose(as.data.frame(storage.num,col.names = colnames(data.in)[i]))
    }
  }
  return(storage.list)
}

test.df<-get.sample.means(data.clustered,11)
sample.means.df<-bind_cols(test.df)

#outputting sample means
write_xlsx(sample.means.df,"C:\\Users\\corne\\Desktop\\test_out.xlsx")
write_xlsx(data.cluster.df,"C:\\Users\\corne\\Desktop\\test_out.xlsx")
summary(as.factor(data.clustered$clara.rf))

#add region in place of state to data
data.dummy<-data.clustered
level.region<-rep(4,10000)
data.dummy<-cbind(data.dummy,level.region)
for (i in 1:10000){
  if (data.dummy[i,10]%in%c("CT","ME","MA","NH","RI","VT","NJ","NY","PA")==TRUE){
    data.dummy[i,13]<-1
  } else if (data.dummy[i,10]%in%c("IL","IN","MI","OH","WI","IA","KS","MN","MO","NE","ND","SD")==TRUE){
    data.dummy[i,13]<-2
  } else if (data.dummy[i,10]%in%c("DE","FL","GA","MD","NC","SC","VA","DC","WV","AL","KY","MS","TN","AR","LA","OK","TX")==TRUE){
    data.dummy[i,13]<-3
  }
}
data.dummy$level.region<-as.factor(data.dummy$level.region)


#start Neural Network process ###########################################################################################################
data.nn<-data.dummy[,-12]

#scale numeric predictors
data.nn$cdc_case_earliest_dt<-scale(data.nn$cdc_case_earliest_dt)
data.nn$cdc_report_dt<-scale(data.nn$cdc_report_dt)
data.nn$onset_dt<-scale(data.nn$onset_dt)

#change sex to 0 for female and 1 for male
nn.sex<-rep(1,nrow(data.nn))
for (i in 1:nrow(data.nn)){
  if (data.nn$sex[i]==levels(data.nn$sex)[1]){
    nn.sex[i]<-0
  }
}
data.nn<-cbind(data.nn,nn.sex)

#change hosp_yn to 1 for yes and 0 for no
nn.hosp<-rep(1,nrow(data.nn))
for (i in 1:nrow(data.nn)){
  if (data.nn$hosp_yn[i]==levels(data.nn$hosp_yn)[1]){
    nn.hosp[i]<-0
  }
}
data.nn<-cbind(data.nn,nn.hosp)

#change death_yn to 1 for yes and 0 for no
nn.death<-rep(1,nrow(data.nn))
for (i in 1:nrow(data.nn)){
  if (data.nn$death_yn[i]==levels(data.nn$death_yn)[1]){
    nn.death[i]<-0
  }
}
data.nn<-cbind(data.nn,nn.death)

#change medcond_yn to 1 for yes and 0 for no
nn.medcond<-rep(1,nrow(data.nn))
for (i in 1:nrow(data.nn)){
  if (data.nn$medcond_yn[i]==levels(data.nn$medcond_yn)[1]){
    nn.medcond[i]<-0
  }
}
data.nn<-cbind(data.nn,nn.medcond)

#change names to format for neuralnet and drop hospital info
data.nn.matrix<-model.matrix(~race_ethnicity_combined+cdc_case_earliest_dt+cdc_report_dt+nn.sex+onset_dt+nn.hosp+nn.medcond+age_group+level.region+nn.death,data = data.nn)
data.nn.matrix<-data.nn.matrix[,-1]
colnames(data.nn.matrix)[1]<-"asian"
colnames(data.nn.matrix)[2]<-"black"
colnames(data.nn.matrix)[3]<-"hispanic"
colnames(data.nn.matrix)[4]<-"multiple"
colnames(data.nn.matrix)[5]<-"nh_pi"
colnames(data.nn.matrix)[6]<-"white"
colnames(data.nn.matrix)[13]<-"ten_nineteen"
colnames(data.nn.matrix)[14]<-"twenty_twentynine"
colnames(data.nn.matrix)[15]<-"thirty_thirtynine"
colnames(data.nn.matrix)[16]<-"forty_fortynine"
colnames(data.nn.matrix)[17]<-"fifty_fiftynine"
colnames(data.nn.matrix)[18]<-"sixty_sixtynine"
colnames(data.nn.matrix)[19]<-"seventy_seventynine"
colnames(data.nn.matrix)[20]<-"eighty_plus"


col_list<-paste(c(colnames(data.nn.matrix[,-24])),collapse = "+")
col_list<-paste(c("nn.death~",col_list),collapse = "")
f<-formula(col_list)

#build the NN model
log.nn<-neuralnet(f,data = data.nn.matrix[train,],hidden = 3,threshold=0.1, act.fct = "logistic",likelihood = TRUE)

#make an observation to get predictions with
test.case<-data.nn.matrix[1:2,-23]

#clear out test case
for (i in 1:22){
  test.case[1,i]<-0
}

#################
#function to set test observation
# numbers to input to get.test function for appropriate individual
# 1 asian 2 black 3 hispanic 6 white
#12 10-19 13 20-29 etc.
#0 NE 20 MW 21 South 22 West
get.test<-function(test.case,race,sex,medcond,age,region){
  for (i in 1:22){
    test.case[1,i]<-0
  }
  test.case[1,race]<-1
  test.case[1,9]<-sex
  test.case[1,11]<-medcond
  test.case[1,age]<-1
  if (region!=0){
    test.case[1,region]<-1
  }
  return(test.case)
}

#loop to get test case prediction over multiple age groups
for (i in 12:19){
  test.case1<-get.test(test.case,6,1,0,i,0)
  #test.case1
  
  test.pred<-compute(log.nn,test.case1)
  print(test.pred$net.result[1])
}


########################clustering exploration###############################################################

data.cluster.explo<-data.nn.matrix

k.means.cluster<-kmeans(data.cluster.explo[,-23],4,nstart = 5)

data.cluster.explo.clustered<-cbind(data.cluster.explo,k.means.cluster$cluster)


#work to get 2d projection
norm_vec<-function(x){
  sqrt(sum(x^2))
}
library(geometry)


get_plot_pt<-function(obs){
  obs.mod<-rep(NA,22)
  for (i in 1:22){
    obs.mod[i]<-obs[i]
  }
  w1<-rep(1,22)
  w2<-rep(0,22)
  w2[c(3,5,7,9)]<-1
  w1hat<-w1/norm_vec(w1)
  w2apo<-w2-dot(w2,w1hat)*w1hat
  w2hat<-w2apo/norm_vec(w2apo)
  plotpt<-c(dot(obs.mod,w1hat),dot(obs.mod,w2hat))
  return(plotpt)
}

two_dim<-setNames(data.frame(matrix(ncol = 3, nrow = 10000)), c("x", "y", "cluster"))

for (i in 1:10000){
  obso<-get_plot_pt(data.cluster.explo[i,])
  two_dim[i,1]<-obso[1]
  two_dim[i,2]<-obso[2]
}

for (i in 1:10000){
  two_dim[i,3]<-k.means.cluster$cluster[i]
}

#density plots for deaths v survivors
deaths.cluster.explo<-which(data.cluster.explo[,23]==1)
surviv.cluster.explo<-(-deaths.cluster.explo)

#death group density plot
p1<-ggplot(data=two_dim[deaths.cluster.explo,],aes(x=two_dim[deaths.cluster.explo,1],y=two_dim[deaths.cluster.explo,2]),bins=100)+geom_bin2d()+scale_fill_gradient(low="lightpink2",high = "darkred")+xlab("")+ylab("")+theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank())+ggtitle("Deaths")+theme(plot.title = element_text(hjust = 0.5))

#survivor group density plot
p2<-ggplot(data=two_dim[surviv.cluster.explo,],aes(x=two_dim[surviv.cluster.explo,1],y=two_dim[surviv.cluster.explo,2]),bins=100)+geom_bin2d()+scale_fill_gradient(low="lightblue2",high = "darkblue")+xlab("")+ylab("")+theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank())+ggtitle("Survivors")+theme(plot.title = element_text(hjust = 0.5))

#density plots for clusters
fviz_cluster(k.means.cluster,two_dim[,-3])+xlab("")+ylab("")+theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank())+ggtitle("Clusters")+theme(plot.title = element_text(hjust = 0.5))

#get percentage of membership in certain feature levels by cluster
agecluster<-plot(x=data.clustered$age_group,y=as.factor(data.cluster.explo.clustered[,24]))
sexcluster<-plot(x=data.clustered$sex,y=as.factor(data.cluster.explo.clustered[,24]))
racecluster<-plot(x=data.clustered$race_ethnicity_combined,y=as.factor(data.cluster.explo.clustered[,24]))
medcondcluster<-plot(x=data.clustered$medcond_yn,y=as.factor(data.cluster.explo.clustered[,24]))




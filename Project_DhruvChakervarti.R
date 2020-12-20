library(readxl)
library(data.table)
library(zoo)
library(fBasics)
library(DescTools)

setwd("C:/Users/idhru/Desktop/QAM/Project/My project")

Industry <- data.table(read.csv('10_Industry_Portfolios.csv',na.strings=c("","NA"),skip=11))
Industry <- Industry[c(1:1126)]

FF_mkt <- data.table(read.csv('F-F_Research_Data_Factors.csv',na.strings=c("","NA"),skip=3))
FF_mkt <- FF_mkt[c(1:1126)]


project <- function(ind_data,ff_data){
  #ind_data <- Industry
  ind_data$Date <- as.Date(paste0(as.character(ind_data$X), '01'), format='%Y%m%d') 
  ind_data$Year <- year(ind_data$Date)
  ind_data$Month <- month(ind_data$Date)
  ind_data <- ind_data[Year>=1989 & Year<=2019,]
  ind_data <- ind_data[9:372,]
  
  #ff_data <- FF_mkt[,c(1,2,5)]
  ff_data <- ff_data[,c(1,2,5)]
  ff_data$Date <- as.Date(paste0(as.character(ff_data$X), '01'), format='%Y%m%d') 
  ff_data$Year <- year(ff_data$Date)
  ff_data$Month <- month(ff_data$Date)
  ff_data <- ff_data[Year>=1989 & Year<=2019,]
  ff_data <- ff_data[9:372,]         
  
  data <- merge(ind_data, ff_data, by=c("X","Date","Year","Month"))
  data <- data[!(data$NoDur == '-99.99' & data$Durbl == '-99.99' & data$Manuf == '-99.99' 
                 & data$Enrgy == '-99.99' & data$HiTec == '-99.99' & data$Telcm == '-99.99'
                 & data$Shops == '-99.99' & data$Hlth == '-99.99' & data$Utils == '-99.99'
                 & data$Other == '-99.99' & data$Mkt.RF == '-99.99' & data$RF == '-99.99' ),]
  data_means <- data[,5:16]
  temp <- data_means[, lapply(.SD, as.character)]
  temp2 <- temp[, lapply(.SD, as.numeric, na.rm = T)]
  temp3 <- temp2[,list(ExNoDur = NoDur - RF, ExDurbl = Durbl - RF, ExManuf = Manuf - RF, ExEnrgy = Enrgy - RF,
                       ExHiTec = HiTec - RF, ExTelcm = Telcm - RF, ExShops = Shops - RF, ExHlth = Hlth - RF,
                       ExUtils = Utils - RF, ExOther = Other - RF, Mkt.RF)]
  
  temp3$Mean_rows <- round(rowMeans(temp3),digit=2)
  
  data2 <- cbind(data[,c("Date","Year","Month")],temp3/100)
  
  #EW  
  mean_total_ew <- mean(data2$Mean_rows)
  sd_total_ew <- sd(data2$Mean_rows)
  Sharpe_Ratio_total_ew <- mean(data2$Mean_rows)/sd(data2$Mean_rows)
  CEQ_total_ew <- mean(data2$Mean_rows) - 0.5*var(data2$Mean_rows)
  turnover <- vector()
  turnover[1] <- 0
  temp_outsample <- data2[,4:14]
  for(i in 2:nrow(temp_outsample)){
      turnover[i] <- sum(abs((1/11) - ((1/11) * (1+temp_outsample[i]))/ (1 + sum((1/11)* temp_outsample[i])))) 
    }
  Turnover_ew <- mean(turnover)
  
  #VW 
  mean_total_vw <- mean(data2$Mkt.RF)
  sd_total_vw <- sd(data2$Mkt.RF)
  Sharpe_Ratio_total_vw <- mean(data2$Mkt.RF)/sd(data2$Mkt.RF)
  CEQ_total_vw <- mean(data2$Mkt.RF) - 0.5*var(data2$Mkt.RF)
  Turnover_vw <- 0
  
  #MVE in sample
  sigma <- cov(data2[,4:14])
  mu <- colMeans(data2[,4:14])     #1 to 120
  xt <- solve(sigma) %*% mu
  weights <- xt/sum(xt)
  ret <- rowSums(weights * data2[,4:14])
  mean_MVE_in <- mean(ret)
  mean_MVE_in
  sd_MVE_in <- sd(ret)
  sd_MVE_in
  Sharpe_MVE_in <- mean_MVE_in/sd_MVE_in
  Sharpe_MVE_in
  CEQ_MVE_in <- mean_MVE_in - 0.5*sd_MVE_in^2
  CEQ_MVE_in
  Turnover_MVE_in <- NaN
  Turnover_MVE_in
  
  
  #MVE Out of sample test
  temp_outsample <- data2[,4:14]
  ret <- data.frame()
  w <- 0
  turnover <- 0
  for(i in 60:nrow(temp_outsample)) {
    sigma <- cov(temp_outsample[(i-59):i])
    mu <- colMeans(temp_outsample[(i-59):i])     #1 to 60
    xt <- solve(sigma) %*% mu
    weights <- xt/sum(xt)
    ret <- rbind(ret,weights * temp_outsample[i+1])
    if((i-59) == 1){
      turnover[i-59] <- 0
    } else {
      turnover[i-59] <- sum(abs(weights - (w * (1+temp_outsample[i]))/ (1 + sum(weights * temp_outsample[i])))) 
    }
    w <- weights
  }
  ret <- ret[1:304,]
  ret_MVE_out <- rowSums(ret)
  mean_MVE_out <- mean(ret_MVE_out)
  mean_MVE_out
  sd_MVE_out <- sd(ret_MVE_out)
  sd_MVE_out
  Sharpe_MVE_out <- mean_MVE_out/sd_MVE_out
  Sharpe_MVE_out
  CEQ_MVE_out <- mean_MVE_out - 0.5*sd_MVE_out^2
  CEQ_MVE_out
  Turnover_MVE_out <- mean(turnover,na.rm=T)
  Turnover_MVE_out
  
  
  #Risk Parity
  temp_outsample <- data2[,4:14]
  ret <- data.frame()
  w <- 0
  turnover <- 0
  for(i in 60:nrow(temp_outsample)) {
    sigma <- cov(temp_outsample[(i-59):i])
    diag_sigma <- diag(sigma)
    xt <- 1/diag_sigma
    weights <- xt/sum(xt)
    ret <- rbind(ret,weights * temp_outsample[i+1])
    if((i-59) == 1){
      turnover[i-59] <- 0
    } else {
      turnover[i-59] <- sum(abs(weights - (w * (1+temp_outsample[i]))/ (1 + sum(weights * temp_outsample[i])))) 
    }
    w <- weights
  }
  ret <- ret[1:304,]
  ret_rp <- rowSums(ret)
  mean_rp <- mean(ret_rp)
  mean_rp
  sd_rp <- sd(ret_rp)
  sd_rp
  Sharpe_rp <- mean_rp/sd_rp
  Sharpe_rp
  CEQ_rp <- mean_rp - 0.5*sd_rp^2
  CEQ_rp
  Turnover_rp <- mean(turnover,na.rm=T)
  Turnover_rp
  
  
  total <- rbind(c(mean_total_ew,mean_total_vw,mean_MVE_in, mean_MVE_out, mean_rp), 
                 c(sd_total_ew,sd_total_vw, sd_MVE_in, sd_MVE_out, sd_rp), 
                 c(Sharpe_Ratio_total_ew, Sharpe_Ratio_total_vw, Sharpe_MVE_in,Sharpe_MVE_out, Sharpe_rp), 
                 c(CEQ_total_ew, CEQ_total_vw, CEQ_MVE_in, CEQ_MVE_out, CEQ_rp), 
                 c(Turnover_ew, Turnover_vw, Turnover_MVE_in, Turnover_MVE_out ,Turnover_rp))
  rownames(total) <- c("Mean","SD","Sharpe Ratio","CEQ","Turnover")
  colnames(total) <- c("EW","VW","MVE Insample","MVE Outsample","RP")
  return(t(total))
}

project(Industry, FF_mkt)

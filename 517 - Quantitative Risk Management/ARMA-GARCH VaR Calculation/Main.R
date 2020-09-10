install.packages("rugarch")
require(rugarch)
options(scipen=999)
options(max.print=2000)

log_ret = read.table("intel_d_logret.txt",header=TRUE)

specGarch11 <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                          mean.model     = list(armaOrder = c(0, 0)))

specAr1Garch11 <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                             mean.model     = list(armaOrder = c(1, 0)))

specArma11Garch11 <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                                mean.model     = list(armaOrder = c(1, 1)),
                                distribution.model = "std")

fitGarch11 <- ugarchfit(spec = specGarch11, data = log_ret$logret)
fitAr1Garch11 <- ugarchfit(spec = specAr1Garch11, data = log_ret$logret)
fitArma11Garch11 <- ugarchfit(spec = specArma11Garch11, data = log_ret$logret)

forecastGarch11 <- ugarchforecast(fitGarch11, n.ahead=10)
forecastAr1Garch11 <- ugarchforecast(fitAr1Garch11, n.ahead=10)
forecastArma11Garch11 <- ugarchforecast(fitArma11Garch11, n.ahead=10)

#$1,000,000 long 99% Value at Risk = e^(conditional mean + conditional standard deviation * 0.99-normal-quantile)-1
#1-Day 99% VaR = $19373.23
1000000*(exp(fitted(forecastGarch11)[1]+sigma(forecastGarch11)[1]*2.333)-1)
#10-Day 99% VaR = $20673.97
1000000*(exp(fitted(forecastGarch11)[10]+sigma(forecastGarch11)[10]*2.333)-1)

#$1,000,000 long 99% Value at Risk = e^(conditional mean + conditional standard deviation * 0.99-normal-quantile)-1
#1-Day 99% VaR = $19381.44
1000000*(exp(fitted(forecastAr1Garch11)[1]+sigma(forecastAr1Garch11)[1]*2.333)-1)
#10-Day 99% VaR = $20621.8
1000000*(exp(fitted(forecastAr1Garch11)[10]+sigma(forecastAr1Garch11)[10]*2.333)-1)

#Student-T degrees of freedom = 6.817979
fitArma11Garch11@fit$coef[7]


#$1,000,000 long 99% Value at Risk = e^(conditional mean + conditional standard deviation * 0.99-student-t-quantile)-1
#1-Day 99% VaR = $22551.09
1000000*(exp(fitted(forecastArma11Garch11)[1]+sigma(forecastArma11Garch11)[1]*qt(0.99,df=fitArma11Garch11@fit$coef[7]))-1)
#10-Day 99% VaR = $22936.34
1000000*(exp(fitted(forecastArma11Garch11)[10]+sigma(forecastArma11Garch11)[10]*qt(0.99,df=fitArma11Garch11@fit$coef[7]))-1)

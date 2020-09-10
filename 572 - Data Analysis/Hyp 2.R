library(car)
options(scipen=999)
fireplaceage = read.table("C:\\Users\\Nick\\Desktop\\AMS\\572\\Project\\AMS 572 Housing Data.csv",
                          header=TRUE, sep=",", colClasses=c("NULL","NULL","NULL","NULL",NA,"NULL",NA))
length(fireplaceage$Fireplaces[fireplaceage$Fireplaces==0])
length(fireplaceage$Fireplaces[fireplaceage$Fireplaces==1])
length(fireplaceage$Fireplaces[fireplaceage$Fireplaces==2])
length(fireplaceage$Fireplaces[fireplaceage$Fireplaces==3])
length(fireplaceage$Fireplaces[fireplaceage$Fireplaces==4])
length(fireplaceage$Fireplaces[fireplaceage$Fireplaces>=5])

fireplaceage$Fireplaces[fireplaceage$Fireplaces==3] = 2
fireplaceage$Fireplaces[fireplaceage$Fireplaces==4] = 2

fireplaceage$Fireplaces[fireplaceage$Fireplaces==0] = '0'
fireplaceage$Fireplaces[fireplaceage$Fireplaces==1] = '1'
fireplaceage$Fireplaces[fireplaceage$Fireplaces==2] = '2'

#raw data analysis
boxplot(fireplaceage$Age~fireplaceage$Fireplaces, names=c("0","1","2+"), ylab="Age", xlab="Fireplaces")

shapiro.test(fireplaceage$Age[fireplaceage$Fireplaces=='0'])
shapiro.test(fireplaceage$Age[fireplaceage$Fireplaces=='1'])
shapiro.test(fireplaceage$Age[fireplaceage$Fireplaces=='2'])
leveneTest(fireplaceage$Age~fireplaceage$Fireplaces)

var(fireplaceage$Age[fireplaceage$Fireplaces=='0'])
var(fireplaceage$Age[fireplaceage$Fireplaces=='1'])
var(fireplaceage$Age[fireplaceage$Fireplaces=='2'])

fit = aov(fireplaceage$Age~fireplaceage$Fireplaces)
summary(fit)
plot(fit)

shapiro.test(fit$residuals)

#log(x+1) transformed data analysis
fireplacelogage = fireplaceage
fireplacelogage$Age = log(fireplacelogage$Age+1)

boxplot(fireplacelogage$Age~fireplacelogage$Fireplaces, names=c("0","1","2+"), ylab="log(Age+1)", xlab="Fireplaces")

shapiro.test(fireplacelogage$Age[fireplacelogage$Fireplaces=='0'])
shapiro.test(fireplacelogage$Age[fireplacelogage$Fireplaces=='1'])
shapiro.test(fireplacelogage$Age[fireplacelogage$Fireplaces=='2'])
leveneTest(fireplacelogage$Age~fireplacelogage$Fireplaces)

var(fireplacelogage$Age[fireplacelogage$Fireplaces=='0'])
var(fireplacelogage$Age[fireplacelogage$Fireplaces=='1'])
var(fireplacelogage$Age[fireplacelogage$Fireplaces=='2'])

fit = aov(fireplacelogage$Age~fireplacelogage$Fireplaces)
summary(fit)
plot(fit)

shapiro.test(fit$residuals)


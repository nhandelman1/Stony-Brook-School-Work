library(car)
options(scipen=999)
fireplaceage = read.table("C:\\Users\\Nick\\Desktop\\AMS\\572\\Project\\AMS 572 Housing Data.csv",
                      header=TRUE, sep=",", colClasses=c("NULL","NULL","NULL","NULL",NA,"NULL",NA))
length(fireplaceage$Fireplaces[fireplaceage$Fireplaces==2])
length(fireplaceage$Fireplaces[fireplaceage$Fireplaces==3])
length(fireplaceage$Fireplaces[fireplaceage$Fireplaces==4])
length(fireplaceage$Fireplaces[fireplaceage$Fireplaces>=5])

fireplaceage$Fireplaces[fireplaceage$Fireplaces==3] = 2
fireplaceage$Fireplaces[fireplaceage$Fireplaces==4] = 2
fireplaceage$Fireplaces[fireplaceage$Fireplaces==0] = '0'
fireplaceage$Fireplaces[fireplaceage$Fireplaces==1] = '1'
fireplaceage$Fireplaces[fireplaceage$Fireplaces==2] = '2'

#raw data
boxplot(fireplaceage$Age~fireplaceage$Fireplaces, names=c("0","1","2+"), ylab="Age", xlab="Fireplaces")

shapiro.test(fireplaceage$Age[fireplaceage$Fireplaces=='0'])
shapiro.test(fireplaceage$Age[fireplaceage$Fireplaces=='1'])
shapiro.test(fireplaceage$Age[fireplaceage$Fireplaces=='2'])

leveneTest(fireplaceage$Age~fireplaceage$Fireplaces)

fit = aov(fireplaceage$Age~fireplaceage$Fireplaces)
summary(fit)

#log(x+1) transformed data

options(scipen=999)
priceage = read.table("C:\\Users\\Nick\\Desktop\\AMS\\572\\Project\\AMS 572 Housing Data.csv",
                      header=TRUE, sep=",", colClasses=c(NA,"NULL","NULL","NULL","NULL","NULL",NA))
median(priceage$Age)

underequal18 = subset(priceage, Age<=18)[,1]
greater18 = subset(priceage, Age>18)[,1]
length(underequal18)
length(greater18)

boxplot(underequal18,greater18, names=c("Age<=18","Age>18"), ylab="Price")

shapiro.test(underequal18)
shapiro.test(greater18)

var.test(underequal18, greater18)

t.test(underequal18, greater18)

plot(priceage[,2],priceage[,1],xlim=c(0, 250), ylim=c(0, 600000),xlab="Age",ylab="Price")

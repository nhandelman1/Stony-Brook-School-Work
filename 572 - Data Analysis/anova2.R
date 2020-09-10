options(scipen=999)
fireplaceage = read.table("C:\\Users\\Nick\\Desktop\\AMS\\572\\Project\\AMS 572 Housing Data.csv",
                          header=TRUE, sep=",", colClasses=c("NULL","NULL","NULL","NULL",NA,"NULL",NA))
ageunderequal11 = subset(fireplaceage, Age<=11)
age11to24 = subset(fireplaceage, Age>11 & Age<=24)
agegreater24 = subset(fireplaceage, Age>24)

fireplaceage$Group[fireplaceage$Age<=11]='0'
fireplaceage$Group[fireplaceage$Age>11 & fireplaceage$Age<=24]='1'
fireplaceage$Group[fireplaceage$Age>24]='2'

hist(fireplaceage$Fireplaces[fireplaceage$Group=='1'])

fireplaceage$Fireplaces[fireplaceage$Group=='2']

??leveneTest

leveneTest(fireplaceage$Fireplaces~fireplaceage$Group)

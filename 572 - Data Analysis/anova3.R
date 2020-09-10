bathroomage = read.table("C:\\Users\\Nick\\Desktop\\AMS\\572\\Project\\AMS 572 Housing Data.csv",
                          header=TRUE, sep=",", colClasses=c("NULL","NULL",NA,"NULL","NULL","NULL",NA))
unique(bathroomage$Bathrooms)

bathroomage$Bathrooms[bathroomage$Bathrooms==4.5]=3.5

length(bathroomage$Bathrooms[bathroomage$Bathrooms==3.5])

boxplot(bathroomage$Age~bathroomage$Bathrooms)


fireplaceage$Group[fireplaceage$Age<=11]='0'
fireplaceage$Group[fireplaceage$Age>11 & fireplaceage$Age<=24]='1'
fireplaceage$Group[fireplaceage$Age>24]='2'

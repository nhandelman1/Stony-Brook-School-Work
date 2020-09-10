options(scipen=999)
fireplaceage = read.table("C:\\Users\\Nick\\Desktop\\AMS\\572\\Project\\AMS 572 Housing Data.csv",
                          header=TRUE, sep=",", colClasses=c(NA,"NULL","NULL","NULL","NULL",NA,"NULL"))
plot(fireplaceage$Lot.Size, fireplaceage$Price)

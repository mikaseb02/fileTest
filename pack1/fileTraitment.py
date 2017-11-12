F2 = open("C:/Users/Mikael/PycharmProjects/fileTest/file.txt","r")

a = F2.readlines()

#print a[0]



#print b

def fileRead(path, delimit, numberColumn) :
    file = open(path,"r")
    path2 = path[:-4] +"2.txt"
    print path2
    f = open(path2, "w")
    test = 1
    nbPrec = 0
    for line in file:
        a = line
        print a
        b = a.split(delimit)
        if len(b) == numberColumn:
            print("OK")
            f.write(a)
        else:
            if test == 1:
                if nbPrec + len(b) < delimit:
                    f.write(a.replace("\n", ""))
                    test = 0
                else:
                    f.write(a)
                    test = 1
            else:
                f.write(delimit + a)
                test = 1
        nbPrec = len(b)
        #file.close()
        #f.close()

fileRead("C:/Users/Mikael/PycharmProjects/fileTest/file.txt", " " , 6)



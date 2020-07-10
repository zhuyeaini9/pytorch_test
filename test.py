def createGenerator():
    mylist = range(3)
    print('kkk')
    for i in mylist:
        print('ppp')
        yield i*i

mygenerator = createGenerator()
print('ggg')
for i in mygenerator:
    print(i)
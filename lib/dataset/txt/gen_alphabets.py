import os


a = 'chars_5990.txt'

s = open(a).readlines()

aa = ''
for ss in s:
    aa += ss.strip()

open('a.txt', 'w').write(aa)
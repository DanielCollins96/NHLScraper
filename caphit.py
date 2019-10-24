import requests
import lxml.html as lh
import pandas as pd
import re

url = 'https://www.spotrac.com/nhl/calgary-flames/cap/'

page = requests.get(url)

doc = lh.fromstring(page.content)

tr_elements = doc.xpath('//tr')
#print(tr_elements)

col=[]
i=0

stats = {
    'name' : 'doe',
    'pos'  : 'field',
    '2' : '',
    '3' : '',
    '4' : ''
 }

for T in tr_elements:
    # print(T.text_content())
    i += 1
    # stats[i] = 
    name= T.text_content()
    name = re.sub(r'(^[ \t]+|[ \t]+(?=:))', '', name, flags=re.M)+','
    # print('%d:"%s"'%(i,name))
    col.append((name.strip(),[]))
 
x = re.search("[^\s]+", col)
if(x):
    print(x)
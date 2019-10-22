from bs4 import BeautifulSoup
import requests
import json
import re


r = requests.get("https://www.spotrac.com/nhl/calgary-flames/cap/")

data = r.text
soup = BeautifulSoup(data, "html.parser")

print('who')
playa = []
print soup.td['player']
for f in soup.find_all('td', 'class'):
    if ( f['class'] == ['player']):
        print(f['class'].children)
    

print(soup.td['class'])

# for p in soup.find_all('tr', attrs={'role' : "row"}):
#     print('penis')
# for weewee in soup.find_all('td', {"class": "player"}):
#     playa.append(weewee[1])
#     print('penis')
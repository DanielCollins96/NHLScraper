from bs4 import BeautifulSoup
import requests
import json
import re


r = requests.get("https://www.spotrac.com/nhl/calgary-flames/cap/")

data = r.text
soup = BeautifulSoup(data, "html.parser")


table = soup.find('table')

table_rows = table.findAll('tr')

for tr in table_rows:
    td = tr.find_all('td')
    row = [i.text for i in td]
    print(row)

# print('who')
# playa = []
# print soup.td['player']
# for f in soup.find_all('td', 'class'):
#     if ( f['class'] == ['player']):
#         print(f['class'].children)
    

# print(soup.td['class'])

# for p in soup.find_all('tr', attrs={'role' : "row"}):
#     print('penis')
# for weewee in soup.find_all('td', {"class": "player"}):
#     playa.append(weewee[1])
#     print('penis')
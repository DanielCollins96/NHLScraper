from bs4 import BeautifulSoup
import requests
import json
import re
from collections import *


# Open the Team Links to get name and logo link in json object it doesnt work right.

tf = open("./Team_Links.txt") 
TeamLinks = tf.readlines() 
# print(TeamLinks)
tf.close()

RosterLinks = []
x = 0
# for p in TeamLinks and x < 3: 
for p in TeamLinks[:3]: 
        #print(p)
        x += 1
        r = requests.get(p)
        data = r.text
        soup = BeautifulSoup(data, "html.parser")
        for player in soup.find_all('td', attrs={'class': 'name-col'}):
                 for link in player.findAll('a', attrs={'href': re.compile("^/")}):
                        print(link)
                        RosterLinks.append("https://www.nhl.com" + link.get('href'))


LogoList = []


# LogoList = []

for p in TeamLinks[:1]:
        r = requests.get(p)
        data = r.text
        soup = BeautifulSoup(data, "html.parser")
        for player in soup.find_all('img', {'class': 'top-nav__club-logo-img'}, {'src':re.compile('.svg')}):
                Logos = {}
                Logos["team"] = soup.title.string.split('|')[1]
                print(soup.title.string.split('|')[1])
                print(player['src'])
                Logos["logo"] = player['src']
                # LogoList = json.dumps(Logos)
                LogoList.append(json.dumps(Logos))
                print(LogoList)


print(LogoList)

from bs4 import BeautifulSoup
import requests
import json
import re
 
r = requests.get("https://www.nhl.com/player/nicolas-deslauriers-8475235")

data = r.text
soup = BeautifulSoup(data, "html.parser")

player = []

nameNum = soup.find('h3', {'class': 'player-jumbotron-vitals__name-num'})
nameNum = nameNum.contents[0]
nameNum = nameNum.split(" | ")
player = nameNum

for stats in soup.findAll('span', {'class':'player-jumbotron-vitals--attr'}):
    #print(stats.contents[0])
    player.append(stats.contents[0])

print(player)
playerDict = dict(name=player[0], number=player[1], position=player[2], height=player[3], weight=player[4], age=player[5], team=player[6])
j = json.dumps(playerDict)
with open('MyRecord,json', 'w') as f:
    f.write(j) 
    f.close()
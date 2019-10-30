from bs4 import BeautifulSoup
import requests
import json
import re
from collections import *
 
r = requests.get("https://www.nhl.com/ducks/roster/2019")

data = r.text
soup = BeautifulSoup(data, "html.parser")

tf = open("./Team_Links.txt") 
TeamLinks = tf.readlines() 
tf.close()


# for page in soup.find_all('ul', attrs={'aria-labelledby': "teamSelect"}):
#     for li in page.findAll('li'):
#         for link in li.findAll('a', attrs={'href': re.compile("^/")}):
#                 TeamLinks.append("https://www.nhl.com" + link.get('href'))

# for p in TeamLinks: 
#         #print(p)
#         r = requests.get(p)
#         data = r.text
#         soup = BeautifulSoup(data, "html.parser")
#         for player in soup.find_all('td', attrs={'class': 'name-col'}):
#                  for link in player.findAll('a', attrs={'href': re.compile("^/")}):
#                         #print(link)
#                         RosterLinks.append("https://www.nhl.com" + link.get('href'))

for p in TeamLinks:
        r = requests.get(p)
        data = r.text
        soup = BeautifulSoup(data, "html.parser")
        for player in soup.find_all('img', {'class': 'top-nav__club-logo-img'}, {'src':re.compile('.svg')}):
                print(player['src'])
                if  player.get('alt'):
                        print(player['alt'])


# teamfile = open("Team_Links.txt", 'w+')
# for item in TeamLinks:
#         teamfile.write(item +"\n")
#         print(item)
# teamfile.close()

# rosterfile = open("Player_Links.txt", 'w+')
# for item in RosterLinks:
#         rosterfile.write(item +"\n")
#         print(item)
# rosterfile.close()

rf = open("./Player_Links.txt")
RosterLinks = rf.readlines()
rf.close()
# print(RosterLinks)
rosters = []

for p in RosterLinks:
        req = requests.get(p)

        data = req.text
        soup = BeautifulSoup(data, "html.parser")

        player = defaultdict(dict)

        #Get The Name
        nameNum = soup.find('h3', {'class': 'player-jumbotron-vitals__name-num'})
        nameNum = nameNum.contents[0]
        nameNum = nameNum.split(" | ")
        #player = nameNum

        player['name'], player['num'] = nameNum[0], nameNum[1]
        
        attributes = []
        #Get the position, height, weight etc..
        for stats in soup.findAll('span', {'class':'player-jumbotron-vitals--attr'}):
                #print(stats.contents[0]+'lol')
                attributes.append((stats.contents[0]))
        player['position'] = attributes[0]
        player['height'] = attributes[1]
        player['weight'] = attributes[2]
        player['age'] = attributes[3].split(' ')[1]
        #Get the Player Picture src
        player_pic = soup.find_all('img', {'class':'player-jumbotron-vitals__headshot-image'}, {'src':re.compile('.jpg')})
        for image in player_pic:
                #print(image['src'])
                player['image'] = (image['src'])

        team_pic = soup.find_all('span', {'class':'player-jumbotron-vitals__team-logo logo-bg-dark--team-24'})
        
        # for index in team_pic: 
        #         print('gottem')


        # print(player)

        playerDict = dict(name=player[0], number=player[1], position=player[2], height=player[3], weight=player[4], age=player[5], team=player[6], image=player[7])
        rosters.append(player)

# j = json.dumps(rosters)
# with open('FlamesRecords.json', 'a+') as f:
#         f.write(j) 
#         f.close()

# {team:'', logo: '', roster: ['']}
import requests
import re
import json
from _collections import deque, defaultdict
from datetime import date

today = date.today()

baseURL = 'https://statsapi.web.nhl.com/api/v1/'

rosters = {}

# with open('./writeTeam.json') as f:
#         d = json.load(f)

# for t in d["teams"][:1]:
#     rosters["team"] = t["name"]
#     print(t["id"])
#     id = t["id"]
#     link = "https://statsapi.web.nhl.com/api/v1/teams/{}/roster".format(id)
#     response = requests.get(link)
#     print(response.text)
# for line in open('./cleanTeams.json', 'r'):
#     print(line)


with open('./data/cleanTeams.json') as f:
    d = json.load(f)

    for t in d["teams"]:
        # id = t
        ts = "{hi}teams/{}/?expand=team.stats".format(t["id"], hi=baseURL)
        response = requests.get(ts)
        r = response.json()

        t["stat"] = r["teams"][0]["teamStats"][0]["splits"][0]["stat"]


d["date"] = today.isoformat()
print(today.isoformat())
with open('cleanteeems2.json', 'w') as jfile:
    json.dump(d, jfile)
    
firebase_data = open('cleanteeems2.json')
response = requests.put('https://hockeydata-e277a.firebaseio.com/teamstats/{dt_time}.json'.format(dt_time=d["date"]), data=firebase_data)


#r["teams"]["teamStats"][0]["splits"][0]
# https://statsapi.web.nhl.com/api/v1/teams/1/?expand=team.stats 



# https://statsapi.web.nhl.com/api/v1/people/8474056



# curl -X PUT -d @cleanLogos.json \
#   'https://hockeydata-e277a.firebaseio.com/teamstats/1.json'





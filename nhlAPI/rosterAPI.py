import requests
import re
import json
from _collections import deque, defaultdict

rosters = {}
baseURL = 'https://statsapi.web.nhl.com/api/v1/'

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
xxx = 0
######### 
with open('./cleanTeams.json') as f:
    d = json.load(f)
for t in d["teams"][:1]:
    # id = t
    ts = "{hi}teams/{}/?expand=team.stats".format(t["id"], hi=baseURL)
    response = requests.get(ts)
    r = response.json()
    for t in r["teams"]:
        # print(t["id"])
        xxx = t["id"]
        for ts in t["teamStats"]:
            for s in ts["splits"]:
                for x in range(len(d["teams"])):
                    if d["teams"][x]["id"] == xxx:
                        # print(d["teams"][x])
                        print((list(s["stat"].keys())))
                        d["teams"][x]["splits"] = s

with open('cleanteeems.json', 'w') as jfile:
    json.dump(d, jfile)

# https://statsapi.web.nhl.com/api/v1/teams/1/?expand=team.stats 



# https://statsapi.web.nhl.com/api/v1/people/8474056





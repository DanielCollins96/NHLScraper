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


with open('./cleanTeams.json') as f:
    d = json.load(f)

    for t in d["teams"]:
        # id = t
        ts = "{hi}teams/{}/?expand=team.stats".format(t["id"], hi=baseURL)
        response = requests.get(ts)
        r = response.json()

        t["stat"] = r["teams"][0]["teamStats"][0]["splits"][0]["stat"]
        # for t in r["teams"]:
        #     # print(t["id"])
        #     id = t["id"]
        #     if id == t["id"]
        #     for ts in t["teamStats"]:
        #         for s in ts["splits"]:
        #         d["teams"][x]["splits"] = ts["splits"][0]
        #             for x in range(len(d["teams"])):
        #                 if d["teams"][x]["id"] == xxx:
        #                     # print(d["teams"][x])
        #                     print((list(s["stat"].keys())[0]))
        #                     d["teams"][x]["splits"] = r["teams"][x]["teamStats"][0]["splits"][0]["stat"]

with open('cleanteeems.json', 'w') as jfile:
    json.dump(d, jfile)

#r["teams"]["teamStats"][0]["splits"][0]
# https://statsapi.web.nhl.com/api/v1/teams/1/?expand=team.stats 



# https://statsapi.web.nhl.com/api/v1/people/8474056





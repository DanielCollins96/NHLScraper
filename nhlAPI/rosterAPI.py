import requests
import re
import json
from _collections import deque, defaultdict

rosters = {}

with open('./writeTeam.json') as f:
        d = json.load(f)

for t in d["teams"][:1]:
    rosters["team"] = t["name"]
    print(t["id"])
    id = t["id"]
    link = "https://statsapi.web.nhl.com/api/v1/teams/{}/roster".format(id)
    response = requests.get(link)
    print(response.text)
# for line in open('./cleanTeams.json', 'r'):
#     print(line)



# print(d.keys())
# for t in d:
#     print(d)






# https://statsapi.web.nhl.com/api/v1/people/8471233





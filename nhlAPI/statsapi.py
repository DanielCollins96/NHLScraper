import requests
import re
import json
from _collections import deque, defaultdict

with open('./data/Teams.json') as f:
        d = json.load(f)
ti = defaultdict(list)
teams = []
# print(d["teams"][:1])
for t in d["teams"]:
    team = {}
    # teams.append(t)
    # print(t)
    team["id"]                  =  t["id"]
    team["franchiseId"]         =  t["franchiseId"]
    team["name"]                =  t["name"]
    team["city"]                =  t["venue"]["city"]
    team["venue"]                =  t["venue"]["name"]
    team["abbreviation"]        =  t["abbreviation"]
    team["firstYearOfPlay"]     =  t["firstYearOfPlay"]
    team["conference"]          =  t["conference"]["name"]
    team["division"]            =  t["division"]["name"]
    team["officialSiteUrl"]     =  t["officialSiteUrl"]
    # print(team)
    teams.append(team)

teams_json = json.dumps(teams)
f = open("writeTeam.json", "w")
f.write(teams_json)
f.close()
# print(teams_json)
# print(teams)

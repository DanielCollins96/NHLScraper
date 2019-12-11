import requests
import re
import json
from _collections import deque, defaultdict


with open('./data/NHL_Teams.json') as f, open('./data/TeamLogos2.json') as l:
        master = json.load(f)
        logos = json.load(l)

    
        for team_logo in logos:
            name = team_logo["team"].strip()
            print(name)
            for team_stats in master["teams"]:
                if name == team_stats["name"]:
                    team_stats["logo"] = team_logo["logo"]
                    print(team_logo["logo"])

        print(master)            
        # with open('cleanLogos.json', 'w') as jfile:
        #     json.dump(master, jfile)
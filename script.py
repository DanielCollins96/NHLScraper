import json
with open('hi.json') as json_data:
    d = json.load(json_data)
print(d.keys())

for key in sorted(d.keys()):
    print(d[key]['name'])
import json

f = 'FlamesRecords.json'
out = 'output.txt'

with open(f, "r") as read_file:
    data = json.load(read_file)
rosterLength = len(data['roster'])
x = 0

for val in range(rosterLength):
    firstName = data['roster'][x]['name'].split(' ')[0]
    lastName = data['roster'][x]['name'].split(' ')[1]
    data['roster'][x]['firstname'].append(firstName)
    data['roster'][x]['lastname'].append(lastName)
    x+=1

# f = open(data)
# json_string = json.dump(f)
# print(f)
# jsondata = json.load(f)
# print(json_string)
read_file.close()
import json

# split_names{'firstname', 'lastname'}
split_names = {}
# input("Enter the filename: ")
path = 'FlamesRecords.json'
outputfile = "./datamang/" + input("Enter filename ") + ".json"

input_file = open(path, 'r')

# with open(path) as f:
#     for line in f:
#         print(json.loads(line))
#     f.close()


# with open(path, 'r') as f:
data = json.load(input_file)

print(len(data['roster']))

rosterSize = len(data['roster'])

x = 0
while x < rosterSize:
    fname = data['roster'][x]["name"].split(" ")[0]
    lname = data['roster'][x]["name"].split(" ")[1]
    data['roster'][x]["firstname"] = fname
    data['roster'][x]["lastname"] = lname
    print(data['roster'][x])

    del data['roster'][x]["team"]
    # print(fname)
    # print(lname)
    # print(data['roster'][x]["name"])
    x += 1

# print(data['roster'][0])
# for x in data:
#     print(data["roster"]["name"])
# print(input_file.read())


with open(outputfile, 'w') as outfile:
    json.dump(data, outfile)
outfile.close()

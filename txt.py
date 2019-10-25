import re

Header = ('Player (22)Pos.AgeBase SalarySigning BonusPerf. BonusTotal SalaryTotal Cap HitAdj. Cap HitCap %,')

DirtyPlayers = [('TkachukMatthew TkachukLW21----$7,000,000 $7,000,000 8.61,'), ('GaudreauJohnny GaudreauLW26$6,750,000--$6,750,000$6,750,000 $6,750,000 8.30,'), ('GiordanoMark GiordanoD35$6,750,000--$6,750,000$6,750,000 $6,750,000 8.30,'), ('MonahanSean MonahanC24$6,750,000--$6,750,000$6,375,000 $6,375,000 7.84,'), ('BacklundMikael BacklundC30$3,850,000$2,000,000-$5,850,000$5,350,000 $5,350,000 6.58,'), ('LucicMilan LucicLW31$3,000,000$3,000,000-$5,250,000$6,000,000 $5,250,000 6.45,'), ('HanifinNoah HanifinD22$4,950,000--$4,950,000$4,950,000 $4,950,000 6.09,'), ('LindholmElias LindholmC24$5,850,000--$5,850,000$4,850,000 $4,850,000 5.96,'), ('BrodieT.J. BrodieD29$4,837,500--$4,837,500$4,650,000 $4,650,000 5.72,'), ('FrolikMichael FrolikRW31$3,000,000--$3,000,000$4,300,000 $4,300,000 5.29,'), ('HamonicTravis HamonicD29$4,875,000--$4,875,000$3,857,143 $3,857,143 4.74,'), ('RyanDerek RyanC32$3,125,000--$3,125,000$3,125,000 $3,125,000 3.84,'), ('RittichDavid RittichG27$2,750,000--$2,750,000$2,750,000 $2,750,000 3.38,'), ('TalbotCameron TalbotG32$2,750,000--$2,750,000$2,750,000 $2,750,000 3.38,'), ('BennettSam BennettC23$2,550,000--$2,550,000$2,550,000 $2,550,000 3.13,'), ('JankowskiMark JankowskiC25$1,750,000--$1,750,000$1,675,000 $1,675,000 2.06,'), ('CzarnikAustin CzarnikC26$1,100,000$150,000-$1,250,000$1,250,000 $1,250,000 1.54,'), ('AnderssonRasmus AnderssonD22$775,000-$57,500$775,000$755,833 $755,833 0.93,'), ('KylingtonOliver KylingtonD22$750,000-$82,500$750,000$730,833 $722,936 0.89,'), ('MangiapaneAndrew MangiapaneLW23$715,000--$715,000$715,000 $715,000 0.88,'), ('RiederTobias RiederRW26$700,000--$700,000$700,000 $700,000 0.86,'), ('StoneMichael StoneD29$700,000--$700,000$700,000 $700,000 0.86,')]

myFile=open('output.txt', 'w')

CleanPlayers = []

index = 0

# re.sub(-{1,})


    shtv = re.search(r'\$[^\]][0-9,]+', p)
    Player['Caphit'] = shtv.group()
    Player['LastName'] = l_name.group()





    CleanPlayers.append(Player)
    # m = re.search(r'-{1,}',p)
    # lol = re.split(r'-{1,}', p)
    print(l_name)
    

for e in CleanPlayers:
    print(e)
myFile.close()



smh = re.search(r'\$[^\]]([^\s]|[a-z])', p)









#!python

############# REGIONS_LCS



def regions_lcs_R(id):
 if id == 0:
  R = 'A'
 elif id == 1:
  R = 'B'
 else:
  R = 'C' 
 return R

def regions_lcs(id,season,arch):

 X1 = 0
 X2 = 0
 Y1 = 0
 Y2 = 0
 R = 'A'
 lsec = 0
 buoys = [0, 0, 0]

##HR

 if arch == 0:
  if season == 0:
   if id == 0:
  # winter
    X1 = 500
    X2 = 800
    Y1 = 180
    Y2 = 480
    R = 'A'
    lsec = 185
    buoys = [80, 120, 150]
   elif id == 1:
    X1 = 610
    X2 = 910
    Y1 = 450
    Y2 = 750
    R = 'B'
    lsec = 150
    buoys = [50, 80, 140]
   elif id == 2:
    X1 = 1200
    X2 = 1500
    Y1 = 170
    Y2 = 470
    R = 'C'
    lsec = 150
    buoys = [75, 150, 225]
  elif season == 1:
   if id == 0:
 # summer
    X1 = 450
    X2 = 750
    Y1 = 200
    Y2 = 500
    R = 'A'
    lsec = 175
    buoys = [80, 120, 160]
   elif id == 1:
    X1 = 650
    X2 = 950
    Y1 = 500
    Y2 = 800
    R = 'B'
    lsec = 180 
    buoys = [80, 110, 150]
   elif id == 2:
    X1 = 1200
    X2 = 1500
    Y1 = 70
    Y2 = 370
    R = 'C'
    lsec = 150
    buoys = [75, 150, 225]
 elif arch == 1: ## LR
  if season == 0:
   if id == 0:
  # winter
    X1 = 730
    X2 = 1030
    Y1 = 110
    Y2 = 410
    R = 'A'
    lsec = 160
    buoys = [80, 120, 150]
   elif id == 1:
    X1 = 700
    X2 = 1000
    Y1 = 490
    Y2 = 790
    R = 'B'
    lsec = 160
    buoys = [80, 120, 150]
   elif id == 2:
    X1 = 1100
    X2 = 1400
    Y1 = 70
    Y2 = 370
    R = 'C'
    lsec = 150
    buoys = [75, 150, 225]
  elif season == 1:
   if id == 0:
 # summer
    X1 = 650
    X2 = 950
    Y1 = 150
    Y2 = 450
    R = 'A'
    lsec = 175
    buoys = [50, 80, 140]
   elif id == 1:
    X1 = 490
    X2 = 790
    Y1 = 430
    Y2 = 730
    R = 'B'
    lsec = 180
    buoys = [70, 110, 150]
   elif id == 2:
    X1 = 1200
    X2 = 1500
    Y1 = 70
    Y2 = 370
    R = 'C'
    lsec = 150
    buoys = [75, 150, 225]

 return X1,X2,Y1,Y2,R,lsec,buoys


## fluidity exp

def regions_fl(time):
 if time == 1:
 # winter box
  X1 = 1300
  X2 = 1330
  Y1 = 265
  Y2 = 320
  R = 'Fs'
 elif time == 0:
 # summer box
  X1 = 1270
  X2 = 1300
  Y1 = 370
  Y2 = 410
  R = 'Fs'
 return X1,X2,Y1,Y2,R


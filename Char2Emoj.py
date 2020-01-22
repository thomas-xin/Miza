import math
import easygui
Char1 = ":RainbowCritterIdle:"
Char2 = ":Critter:"
box = [31,31,31,31,31]
space = [0,0,0,0,0]
exclamation = [2,2,2,0,2]
doublequote = [5,5,0,0,0]
hashtag = [10,31,10,31,10]
dollar = [7,10,6,5,14]
percent = [5,1,2,4,5]
andsign = [4,10,4,10,7]
quote = [2,2,0,0,0]
lbracket = [2,4,4,4,2]
rbracket = [2,1,1,1,2]
asterisk = [21,14,4,14,21]
plus = [0,2,7,2,0]
comma = [0,0,3,3,4]
minus = [0,0,7,0,0]
period = [0,0,3,3,0]
fslash = [1,1,2,4,4]
zero = [7,5,5,5,7]
one = [3,1,1,1,1]
two = [7,1,7,4,7]
three = [7,1,7,1,7]
four = [5,5,7,1,1]
five = [7,4,7,1,7]
six = [7,4,7,5,7]
seven = [7,5,1,1,1]
eight = [7,5,7,5,7]
nine = [7,5,7,1,7]
at = [14,17,17,17,14]
A_ = [2,5,7,5,5]
B_ = [6,5,7,5,6]
C_ = [3,4,4,4,3]
D_ = [6,5,5,5,6]
E_ = [7,4,7,4,7]
F_ = [7,4,7,4,4]
G_ = [7,4,5,5,7]
H_ = [5,5,7,5,5]
I_ = [7,2,2,2,7]
J_ = [7,1,1,5,7]
K_ = [5,5,6,5,5]
L_ = [4,4,4,4,7]
M_ = [17,27,21,17,17]
N_ = [9,13,15,11,9]
O_ = [2,5,5,5,2]
P_ = [7,5,7,4,4]
Q_ = [4,10,10,10,5]
R_ = [6,5,7,6,5]
S_ = [3,4,7,1,6]
T_ = [7,2,2,2,2]
U_ = [5,5,5,5,7]
V_ = [5,5,5,5,2]
W_ = [17,17,21,21,10]
X_ = [5,5,2,5,5]
Y_ = [5,5,2,2,2]
Z_ = [7,1,2,4,7]
symbols1 = [space,exclamation,doublequote,hashtag,dollar,percent,andsign,quote,lbracket,rbracket,asterisk,plus,comma,minus,period,fslash]
numbers = [zero,one,two,three,four,five,six,seven,eight,nine]
letters = [at,A_,B_,C_,D_,E_,F_,G_,H_,I_,J_,K_,L_,M_,N_,O_,P_,Q_,R_,S_,T_,U_,V_,W_,X_,Y_,Z_]
def convertString(string,C_1,C_2):
    string = string.replace("_"," ")
    if C_1 != "":
        Char1 = C_1
    if C_2 != "":
        Char2 = C_2
    printed = ["","","","","","",""]
    for index in range(0,len(string)):
        curr = string[index]
        delta = ord(curr)
        try:
            if delta < 48:
                dat = symbols1[delta-32]
            else:
                num = delta-48
                if num <= 9:
                    dat = numbers[num]
                else:
                    dat = letters[num-16]
        except:
            dat = box
        maxi = max(dat)
        if maxi < 1:
            maxi = 1
        limit = math.trunc(math.log(maxi,2))+1
        if limit < 3:
            limit = 3
        printed[0] += Char2*(limit+1)
        printed[6] += Char2*(limit+1)
        if len(dat) == 5:
            for yat in range(0,5):
                printed[yat+1] += Char2
                for power in range(0,limit):
                    if dat[yat]&2**(limit-1-power):
                        printed[yat+1] += Char1
                    else:
                        printed[yat+1] += Char2
        for xat in range(0,len(printed)):
            printed[xat] += Char2
    output = ""
    for order in range(0,len(printed)):
        output += str(printed[order]) + "\n"
    return output
if __name__ == "__main__":
    while True:
        string = easygui.enterbox(msg='Please input a string: ',title='Text',default='HELLO WORLD',strip=True)
        if string == None:
            break
        C_1 = easygui.enterbox(msg='Please input text format: ',title='Colour1',default=':RainbowCritterIdle:',strip=True)
        C_2 = easygui.enterbox(msg='Please input background format: ',title='Colour2',default=':Critter:',strip=True)
##    string = input("Please input a string: ")
##    C_1 = input("Please input text format: ")
##    C_2 = input("Please input background format: ")
        output = convertString(string.upper(),C_1,C_2)
        if easygui.codebox(msg="Here is your converted text: ",title="Output",text=output) == None:
            break

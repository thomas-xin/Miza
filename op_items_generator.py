import sys
from smath import romanNumerals
def upperword(word):
    output = ""
    negative = False
    if word == "sweeping":
        output = "Sweeping Edge"
    elif word == "bane_of_arthropods":
        output = "Bane of Arthropods"
    elif word == "binding_curse":
        output = "Curse of Binding"
        negative = True
    elif word == "vanishing_curse":
        output = "Curse of Vanishing"
        negative = True
    else:
        for i in range(0,len(word)):
            if word[i] == "_":
                output += " "
            elif word[i-1] == " " or word[i-1] == "_" or i == 0:
                output += word[i].upper()
            else:
                output += word[i].lower()
    return [negative,output]
def generateEnchant(item,enchants):
    command = "/give @p "+item+"{"
    start = """display:{Lore:["""
    middle = """]},HideFlags:1,Enchantments:["""
    end = """]}"""
    lore = ""
    enchant = ""
    for e in enchants:
        level = int(enchants[e])
        if level > 2147483647:
            level = 2147483647
        word = upperword(e)
        if word[0]:
            lore += """'{"text":\""""+word[1]+" "+romanNumerals(level)+"""\","color":"red","italic":false}',"""
        else:
            lore += """'{"text":\""""+word[1]+" "+romanNumerals(level)+"""\","color":"gray","italic":false}',"""
        enchant += "{id:"+e.lower()+",lvl:"+str(level)+"},"
    output = command+start+lore+middle+enchant+end
    return output

if __name__ == "__main__":
    print("""Welcome to the Minecraft OP enchant generator!
    Enter corresponding details to the following questions to generate your command,
    or enter a blank line at any point to stop.
    """)
    while True:
        item = input("Enter name of item to produce: ")
        if item == "":
            break
        enchants = {}
        while True:
            enchantment = input("Enter name of enchantment to add: ")
            if enchantment == "":
                break
            level = input("Enter level of enchantment to add: ")
            if level == "":
                break
            level = int(level)
            enchants[enchantment] = level
        print("Generating command...")
        output = generateEnchant(item,enchants)
        print(output)
        

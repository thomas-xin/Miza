def roman_numerals(num, order=0):
	num = num if type(num) is int else int(num)
	carry = 0
	over = ""
	sym = ""
	output = ""
	if num >= 4000:
		carry = num // 1000
		num %= 1000
		over = roman_numerals(carry, order + 1)
	while num >= 1000:
		num -= 1000
		output += "M"
	if num >= 900:
		num -= 900
		output += "CM"
	elif num >= 500:
		num -= 500
		output += "D"
	elif num >= 400:
		num -= 400
		output += "CD"
	while num >= 100:
		num -= 100
		output += "C"
	if num >= 90:
		num -= 90
		output += "XC"
	elif num >= 50:
		num -= 50
		output += "L"
	elif num >= 40:
		num -= 40
		output += "XL"
	while num >= 10:
		num -= 10
		output += "X"
	if num >= 9:
		num -= 9
		output += "IX"
	elif num >= 5:
		num -= 5
		output += "V"
	elif num >= 4:
		num -= 4
		output += "IV"
	while num >= 1:
		num -= 1
		output += "I"
	if output != "":
		if order == 1:
			sym = "ᴍ"
		elif order == 2:
			sym = "ᴍᴹ"
	return over + output + sym

def unroman_numerals(num):
	values = dict(
		I=1,
		V=5,
		X=10,
		L=50,
		C=100,
		D=500,
		M=1000,
	)
	order = "".join(sorted(values, key=values.get))
	output = []
	num = list(num.upper())
	value = 0
	curr = 0
	c = ""
	while num:
		n = num.pop(0)
		if n == "ᴍ":
			if num[0] == "ᴹ":
				num.pop(0)
				mult = 1000000
			else:
				mult = 1000
			value += curr
			value *= mult
			output.append(value)
			value = 0
			curr = 0
			c = ""
			continue
		if n == c:
			curr += values[n]
			continue
		if c and values[n] > values[c]:
			curr = values[n] - curr
			continue
		value += curr
		curr = values[n]
		c = n
	value += curr
	output.append(value)
	return sum(output)

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
		for i in range(0, len(word)):
			if word[i] == "_":
				output += " "
			elif word[i - 1] == " " or word[i - 1] == "_" or i == 0:
				output += word[i].upper()
			else:
				output += word[i].lower()
	return [negative, output]

def generate_enchant(item, args):
	romans = "IVXLCDMᴍᴹ"
	enchants = {}
	enchant = ""
	for a in args:
		if not enchant:
			if a.endswith(","):
				enchants[a[:-1]] = 1
			else:
				enchant = a
			continue
		if a.endswith(","):
			a = a[:-1]
		if a.isnumeric():
			enchants[enchant] = int(a)
		elif not a.upper().translate("".maketrans({c: "" for c in romans})):
			enchants[enchant] = unroman_numerals(a)
		else:
			enchants[enchant] = 1
		enchant = ""
	if enchant:
		enchants[enchant] = 1
	if not enchants:
		return f"/give @s {item}"
	command = "/give @s "+item+"{"
	start = """display:{Lore:["""
	middle = """]},HideFlags:1,Enchantments:["""
	end = """]}"""
	lore = ""
	enchant = ""
	for e in enchants:
		level = elevel = min(65535, int(enchants[e]))
		if elevel > 255:
			n = elevel // 255
			elevel -= 255 * n
			enchant += ("{id:"+e.lower()+",lvl:255},") * n
		word = upperword(e)
		if word[0]:
			lore += """'{"text":\""""+word[1]+" "+roman_numerals(level)+"""\","color":"red","italic":false}',"""
		else:
			lore += """'{"text":\""""+word[1]+" "+roman_numerals(level)+"""\","color":"gray","italic":false}',"""
		enchant += "{id:"+e.lower()+",lvl:"+str(elevel)+"},"
	output = command+start+lore+middle+enchant+end
	return output
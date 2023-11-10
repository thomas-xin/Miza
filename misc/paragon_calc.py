import sys, requests
from sympy import sympify
from math import *


if __name__ == "__main__":
	ARGS = sys.argv[1:]
	if not ARGS:
		raise SystemExit
else:
	ARGS = None

COSTS = {}

def get_tower_data():
	resp = requests.get("https://bloons.fandom.com/wiki/Module:PricesBTD6/data?action=edit")
	resp.raise_for_status()
	s = resp.text
	s = s.split("--&lt;nowiki>", 1)[1]
	s = s.split("--&lt;/nowiki>", 1)[-2]
	lines = s.splitlines()
	while lines:
		line = lines.pop(0)
		search = "prices["
		if not line.startswith(search):
			continue
		name = line[len(search) + 1:].split('"', 1)[0]
		data = COSTS.setdefault(name, {})
		top = lines.pop(0).split("{", 1)[-1].split("}", 1)[0].split(",")
		middle = lines.pop(0).split("{", 1)[-1].split("}", 1)[0].split(",")
		bottom = lines.pop(0).split("{", 1)[-1].split("}", 1)[0].split(",")
		data["base"] = int(lines.pop(0).split("{", 1)[-1].split("}", 1)[0])
		if not lines or lines[0].strip().startswith("MK = "):
			data["paragon"] = 0
		else:
			data["paragon"] = int(lines.pop(0).split("{", 1)[-1].split("}", 1)[0].strip() or 0)
		data["upgrades"] = [list(map(int, top)), list(map(int, middle)), list(map(int, bottom))]
	return COSTS
get_tower_data()

C2 = {}
def tower_cost(tower, path, village=False, temple=False):
	t = (tower, path, village, temple)
	try:
		return C2[t]
	except KeyError:
		pass
	data = COSTS[tower]
	if tower == "Dart Monkey" and village:
		cost = 0
	else:
		cost = data["base"]
	upgrades = data["upgrades"]
	for i, n in enumerate(path):
		for k in range(0, int(n)):
			c = upgrades[i][k]
			if village and k <= 3:
				c *= 0.85
			cost += c
	if temple:
		cost *= 0.8
	C2[t] = cost
	return cost

def tier_count(t):
	return sum(map(int, t))

def total_power(cost, cash, injected, pops, tier5, tiers, totems):
	return (
		min(floor((cash + injected / 1.05) / cost * 20000), 60000) + min(floor(pops / 180), 90000)
		+ min(tier5 * 6000, 50000) + min(tiers * 100, 10000)
		+ totems * 2000
	)

def max_cost(cost):
	return ceil(cost / 20000 * 60000)

def available_inject(cost, cash):
	return ceil((cost / 20000 * 60000 - cash) * 1.05)

def required_inject(cost, cash, power, required):
	avail = available_inject(cost, cash)
	avapwr = ceil(avail / 1.05 / cost * 20000)
	tarpwr = min(required - power, avapwr)
	return max(0, ceil(tarpwr / 20000 * cost * 1.05))

def d2p(d):
	if d <= 1:
		return 0
	if d >= 100:
		return 200000
	return round((50*d**3+5025*d**2+168324*d+843000)/600)

def p2d(p):
	if p < 2000:
		return 1
	elif p >= 200000:
		return 100
	w = sympify("(-162*p + sqrt((-324*p - 55961091/100)**2 - 531441/250000)/2 - 55961091/200)**(1/3)").subs("p", p)
	return min(max(floor(abs(sympify("-(-1/2 + sqrt(3)*I/2)*w/3 - 67/2 - 27/(100*(-1/2 + sqrt(3)*I/2)*w)").subs("w", w).evalf(24))), 1), 99)

aliases = {
	"Dart Monkey": [],
	"Boomerang Monkey": ["boomer"],
	"Bomb Shooter": ["bomb tower", "cannon"],
	"Tack Shooter": [],
	"Ice Monkey": ["ice tower"],
	"Glue Gunner": [],
	"Sniper Monkey": [],
	"Monkey Sub": ["sub monkey", "submarine"],
	"Monkey Buccaneer": ["buccaneer monkey", "boat"],
	"Monkey Ace": ["ace monkey", "ace tower", "plane"],
	"Heli Pilot": ["heli monkey", "heli tower", "helicopter"],
	"Mortar Monkey": ["mortar tower"],
	"Dartling Gunner": ["dartling monkey", "dartling tower"],
	"Wizard Monkey": ["wizard tower", "wiz"],
	"Super Monkey": [],
	"Ninja Monkey": [],
	"Alchemist": ["alch"],
	"Druid": ["druid monkey"],
	"Banana Farm": ["farm"],
	"Spike Factory": ["spactory", "spac"],
	"Monkey Village": [],
	"Engineer Monkey": ["engi"],
}
alias_map = {}
for k, v in aliases.items():
	for n in v:
		if " " in n:
			for r in ("-", "_", ""):
				alias_map[n.replace(" ", r)] = k
		alias_map[n] = k
	v = k.casefold()
	alias_map[v] = k
	if " " in v:
		for r in ("-", "_", ""):
			alias_map[v.replace(" ", r)] = k
		r = v.split(None, 1)
		if r[0] == "monkey":
			v = r[1]
		else:
			v = r[0]
		alias_map[v] = k
# print(alias_map)

M_CASH = 250000
M_POPS = 16200000
M_TIER5 = 9
M_TIERS = 100

def parse(args):
	if "-v" in args:
		verbose = True
		args.remove("-v")
	else:
		verbose = False
	output = []

	cash = 0
	pops = 0
	generated = 0
	injected = 0
	tier5 = 0
	tiers = 0
	totems = 0
	limit = 34

	t5s = set()
	t = args.pop(0)
	try:
		tower = alias_map[t.casefold()]
	except KeyError:
		raise FileNotFoundError(f'Tower "{t}" not found.')
	if "temple" in args:
		args.remove("temple")
		temple = True
	else:
		temple = False
	if "village" in args:
		args.remove("village")
		village = True
	else:
		village = False
	tcost = lambda t: tower_cost(tower, t, temple=temple, village=village)
	t5cost = []

	for arg in args:
		arg = arg.rstrip(",")
		if arg.endswith("kp"):
			pops += float(arg[:-2]) * 1e3
		elif arg.endswith("mp"):
			pops += float(arg[:-2]) * 1e6
		elif arg.endswith("gp"):
			pops += float(arg[:-2]) * 1e9
		elif arg.endswith("tp"):
			pops += float(arg[:-2]) * 1e12
		elif arg.endswith("p"):
			pops += float(arg[:-1])
		elif arg.endswith("g"):
			pops += float(arg[:-1]) * 4
		elif arg.endswith("t"):
			totems += int(arg[:-1])
		elif arg.endswith("l"):
			limit = min(limit, int(arg[:-1]))
		else:
			if "x" in arg or "*" in arg:
				n, arg = arg.replace("*", "x").split("x", 1)
				n = int(n)
			else:
				n = 1
			t5s.add("".join("5" if c == "5" else "0" for c in arg))
			t5 = arg.count("5")
			if not t5:
				tiers += sum(map(int, arg)) * n
				cash += tcost(arg) * n
			else:
				tier5 += t5 * n
				t5cost.extend([tcost(arg)] * n)
	pcost = apcost = COSTS[tower].get("paragon", 0)
	added = 0
	for cp in ("500", "050", "005"):
		if cp not in t5s:
			added += 1
			tier5 += 1
			c = tcost(cp)
			apcost += c
			t5cost.append(c)
	if added:
		plural = "towers have" if added != 1 else "tower has"
		output.append(f"`Note: The missing `{added}` tier-5 {plural} automatically been added.`")
	output.append(f"Current paragon cost: `{ceil(apcost)}`")
	pcost = pcost or 1000000 / 3
	tier5 = max(0, tier5 - 3)
	M_CASH = max_cost(pcost)
	if tier5 > 0:
		cash += sum(sorted(t5cost)[3:])
		# tiers += tier5 * 4
	pops = floor(pops + generated * 4)
	output.append(f"Current effective cash: `{ceil(cash)}`")
	output.append(f"Current effective pops (p/g): `{pops}`")
	output.append(f"Current tier5 count: `{tier5}`")
	output.append(f"Current upgrade count: `{tiers}`")
	output.append(f"Current totem count (t): `{totems}`")
	power = total_power(pcost, cash, injected, pops, tier5, tiers, totems)
	output.append(f"Current power: `{power}`")
	p = power
	d = p2d(p)
	degree = d
	output.append(f"Current degree: `{degree}`")
	degs = [20, 40, 60, 80, 100]
	while degs:
		n = degs[0]
		if n > d + 1:
			break
		degs.pop(0)
	degs.insert(0, d + 1)
	pows = [d2p(d) for d in degs]
	adds = [p - power for p in pows]

	crosspaths = ("100", "010", "001", "110", "101", "011", "200", "210", "201", "120", "021", "102", "012", "220", "202", "022")
	ccosts = [tcost(cp) for cp in crosspaths]
	effs = [tier_count(cp) / cc for cp, cc in zip(crosspaths, ccosts)]
	order = sorted(range(len(effs)), key=lambda i: effs[i], reverse=True)
	lc_crosspaths = [crosspaths[i] for i in order]
	lc_ccosts = [ccosts[i] for i in order]
	# print(lc_crosspaths, lc_ccosts)

	order = sorted(range(len(effs)), key=lambda i: ccosts[i], reverse=False)
	ll_crosspaths = [crosspaths[i] for i in order]
	ll_ccosts = [ccosts[i] for i in order]

	crosspaths = ("220", "202", "022", "320", "302", "230", "032", "203", "023")
	ccosts = [tcost(cp) for cp in crosspaths]
	effs = [tier_count(cp) / cc for cp, cc in zip(crosspaths, ccosts)]
	order = sorted(range(len(effs)), key=lambda i: effs[i], reverse=True)
	mt_crosspaths = [crosspaths[i] for i in order]
	mt_ccosts = [ccosts[i] for i in order]
	# print(mt_crosspaths, mt_ccosts)

	crosspaths = ("420", "402", "240", "042", "204", "024", "320", "302", "230", "032", "203", "023")
	ccosts = [tcost(cp) for cp in crosspaths]
	order = sorted(range(len(ccosts)), key=lambda i: ccosts[i], reverse=False)
	lt_crosspaths = [crosspaths[i] for i in order]
	lt_ccosts = [ccosts[i] for i in order]
	# print(lt_crosspaths, lt_ccosts)
	stats = [cash, injected, pops, tier5, tiers, totems]
	original = degree
	achieved = original
	for d, p, a in zip(degs, pows, adds):
		if achieved >= d:
			continue
		cash, injected, pops, tier5, tiers, totems = stats
		atowers = []
		if tiers < M_TIERS:
			tc = M_TIERS - tiers
			while tc > 0 and power < p:
				broken = False
				for cp, cc in zip(lc_crosspaths, lc_ccosts):
					t = tier_count(cp)
					if t <= tc and limit * t >= M_TIERS:
						broken = True
						break
				if limit * t < M_TIERS:
					for cp, cc in zip(mt_crosspaths, mt_ccosts):
						t = tier_count(cp)
						if t <= tc and limit * t >= M_TIERS:
							broken = True
							break
				if not broken:
					if t < 5:
						for cp, cc in zip(lc_crosspaths, lc_ccosts):
							if tier_count(cp) >= t:
								break
					else:
						for cp, cc in zip(mt_crosspaths, mt_ccosts):
							if tier_count(cp) >= t:
								break
				atowers.append(cp)
				t = tier_count(cp)
				tc -= t
				cash += cc
				tiers += t
				power = total_power(pcost, cash, injected, pops, tier5, tiers, totems)
		if cash < M_CASH:
			tc = M_CASH - cash
			while tc > 0 and power < p:
				cc = lt_ccosts[-1]
				if cc < tc:
					for cp, cc in zip(lt_crosspaths, lt_ccosts):
						if cc >= tc:
							break
				else:
					cp = lt_crosspaths[-1]
				atowers.append(cp)
				t = tier_count(cp)
				tc -= cc
				cash += cc
				tiers += t
				power = total_power(pcost, cash, injected, pops, tier5, tiers, totems)
		atowers.sort(key=lambda t: tcost(t))
		while atowers and (len(atowers) > limit or power < p and (tiers < M_TIERS or cash < M_CASH)):
			cp = atowers.pop(0)
			cc = tcost(cp)
			t = tier_count(cp)
			cash -= cc
			tiers -= t
			power = total_power(pcost, cash, injected, pops, tier5, tiers, totems)
			if power < p and len(atowers) < limit:
				d = p2d(power)
				req = d2p(d + 1)
				op = cp
				for cp2, cc2 in zip(reversed(lt_crosspaths), reversed(lt_ccosts)):
					if cp2 == op:
						break
					if t >= 6 or tiers + tier_count(cp2) >= M_TIERS or cash + cc2 >= M_CASH:
						cp, cc = cp2, cc2
						break
					elif tier_count(cp2) < t:
						continue
					else:
						cp, cc = cp2, cc2
				# print(atowers, op, cp)
				atowers.append(cp)
				cash += cc
				tiers += tier_count(cp)
				if op == cp:
					power = total_power(pcost, cash, injected, pops, tier5, tiers, totems)
					if power >= req:
						break
					atowers.pop(-1)
					cash -= cc
					tiers -= tier_count(cp)
					break
		atowers.sort(key=lambda t: tcost(t))
		power = total_power(pcost, cash, injected, pops, tier5, tiers, totems)
		if atowers:
			cp = atowers[0]
			temp = (atowers.copy(), cash, pops, tier5, tiers, totems)
			if tiers >= M_TIERS + tier_count(cp) or cash >= M_CASH + tcost(cp):
				rp = power
				if rp > p:
					rp = p
				removed = 0
				while atowers and len(atowers) < limit:
					cp = atowers[0]
					t = tier_count(cp)
					c = tcost(cp)
					estp = total_power(pcost, injected, cash - c, pops, tier5, tiers - t, totems)
					atowers.pop(0)
					power = estp
					cash -= c
					tiers -= t
					if estp < rp:
						if removed > 1 or tiers < M_TIERS:
							break
						removed += 1
				while power < rp:
					cp, cc = ll_crosspaths[0], ll_ccosts[0]
					# for cp, cc in zip(ll_crosspaths, ll_ccosts):
						# if sum(map(int, cp)) < tc:
							# break
					atowers.append(cp)
					t = tier_count(cp)
					tc -= t
					cash += cc
					tiers += t
					power = total_power(pcost, cash, injected, pops, tier5, tiers, totems)
				if len(atowers) > limit:
					atowers, cash, pops, tier5, tiers, totems = temp
					power = total_power(pcost, cash, injected, pops, tier5, tiers, totems)
				else:
					atowers.sort(key=lambda t: tcost(t))
		asacs = []
		curr = ""
		count = 0
		for t in reversed(atowers):
			if curr and t != curr:
				if count > 1:
					asacs.append(f"{count}x{curr}")
				else:
					asacs.append(curr)
				count = 0
			curr = t
			count += 1
		if count > 1:
			asacs.append(f"{count}x{curr}")
		elif curr:
			asacs.append(curr)
		if cash < M_CASH and power < p:
			# injected = available_inject(pcost, cash)
			injected = required_inject(pcost, cash, power, p)
			power = total_power(pcost, cash, injected, pops, tier5, tiers, totems)
		s = " ".join(asacs) if len(asacs) != 1 or "x" in asacs[0] else "(1x)" + asacs[0]
		if power >= p:
			d2 = p2d(power)
			if d2 > d:
				d = d2
				a = d2p(d)
			output.append(f"**Power required for degree {d}: `{a}`**")
			if asacs:
				output.append(f"Recommended additional sacrifices: `{s}`")
			# if verbose:
			ac = cash - stats[0]
			if ac:
				output.append(f"Sacrifice cost: `{ceil(ac)}`")
			ac = injected - stats[1]
			if ac:
				output.append(f"Injected cash: `{ceil(ac)}`")
			ap = pops - stats[2]
			if ap:
				output.append(f"Additional pops: `{ap}`")
			at = tier5 - stats[3]
			if at:
				output.append(f"Additional tier5s: `{at}`")
			au = tiers - stats[4]
			if au:
				output.append(f"Additional upgrades: `{au}`")
			at = totems - stats[5]
			if at:
				output.append(f"Additional totems: `{at}`")
			achieved = d
		else:
			d = p2d(power)
			if achieved >= d:
				search = "**Power required for degree "
				for i, o in enumerate(reversed(output)):
					k = len(output) - i - 1
					if o.startswith(search):
						output[k] = o.replace(":", " (maximum available):", 1)
						break
				break
			p = d2p(d)
			if injected:
				power = total_power(pcost, cash, 0, pops, tier5, tiers, totems)
				injected = required_inject(pcost, cash, power, p)
				p = total_power(pcost, cash, injected, pops, tier5, tiers, totems)
			output.append(f"**Power required for degree {d} (maximum available): `{p}`**")
			if asacs:
				output.append(f"Recommended additional sacrifices: `{s}`")
			# if verbose:
			ac = cash - stats[0]
			if ac:
				output.append(f"Sacrifice cost: `{ceil(ac)}`")
			ac = injected - stats[1]
			if ac:
				output.append(f"Injected cash: `{ceil(ac)}`")
			ap = pops - stats[2]
			if ap:
				output.append(f"Additional pops: `{ap}`")
			at = tier5 - stats[3]
			if at:
				output.append(f"Additional tier5s: `{at}`")
			au = tiers - stats[4]
			if au:
				output.append(f"Additional upgrades: `{au}`")
			at = totems - stats[5]
			if at:
				output.append(f"Additional totems: `{at}`")
			break
	if achieved == original:
		output.append("**No possible degree increases found.**")
	return "\n".join(output)

if ARGS:
	print(parse(ARGS))
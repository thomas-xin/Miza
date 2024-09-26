from misc import util
func = util.tlen
spl = util.split_across("test " * 1000000, lim=12000, func=func)
print(list(map(func, spl)))
print(sum(map(len, spl)))
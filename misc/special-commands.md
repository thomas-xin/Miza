# Special Command Documentation


## Math
> ~math utilises the [SymPy](https://www.sympy.org) library running in concurrent subprocesses to perform calculations. It supports almost all SymPy features, making it an incredibly powerful command. By default it has a timeout limit of 24 seconds at which point subprocesses are automatically restarted, in order to protect the bot against malicious uses of the command. The numerical inputs of many other commands optionally accept any equation that may be evaluated to a real number via this command.

### Examples

~math 1+1
```
1+1 = 2
```


- Evaluates operations in the typical BODMAS order.

~math (4 + 101) * 3.2 ^ 9 / 0.3
```
(4 + 101) * 3.2 ^ 9 / 0.3 = 12314530.2310912
962072674304
────────────
   78125
```


- "deg" is a hardcoded variable that is always equal to π/180 (since SymPy uses radians as the default angle unit)

~math tan(70deg) * 6
```
tan(70deg) * 6 = 16.48486451672773367256998415898603630651123559502495492903163801511938952328955910604362547570421115759740539384257944985532259
     ⎛7∙π⎞
6∙tan⎜───⎟
     ⎝ 18⎠
```


- This wraps the sympy.limit function, allowing easy evaluation of limits as well as substitutions.

~math lim(sin(x) / tan(x), x=0)
```
lim(sin(x) / tan(x), x=0) = 1
```


- Normally this function is called "integrate" in SymPy, "intg" is simply an alias specific to Miza.

~math intg(8/x - 9^x)
```
intg(8/x - 9^x) = -0.45511961331341869680712008286805350030631802862760587236315103164764054159689687332363589041904357414485064160234702996333541748*9.0**x + 8.0*log(x)
   x
- 9  + log(43046721)∙log(x)
───────────────────────────
          2∙log(3)
```


- A space after a function will cause it to operate on as much of the rest of the equation as possible.

~math series cot(x)
```
series cot(x) = 1/x - 0.33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333*x - 0.022222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222*x**3 - 0.0021164021164021164021164021164021164021164021164021164021164021164021164021164021164021164021164021164021164021164021164021164021*x**5 + O(x**6)
         3      5
1   x   x    2∙x     ⎛ 6⎞
─ - ─ - ── - ──── + O⎝x ⎠
x   3   45   945
```


- SymPy is unable to find very large prime factors; Miza uses an implementation of [ECM](https://www.alpertron.com.ar/ECM.HTM) in order to perform factorization of numbers 2^64 or higher.

~math factorize(1201353867969769697054927471068963908362569496107926782221031)
```
factorize(1201353867969769697054927471068963908362569496107926782221031) = [70517684572800905866524262709, 17036206949329404010906952706859]
```


- Able to use all SymPy plotting functions, automatically uploading them to Discord as message attachments.

~math plot atan(x)

![plot_atan](https://cdn.discordapp.com/attachments/320915703102177293/815403492373299200/1614477739417566.png)

- Some additional custom functions:
~math random(1, 6)
```
random(1, 6) = 6
```


~math brainfuck(--[------+<]-----.[----+<]-----.+++.+++[--+++<]-.)
```
brainfuck(--[------+<]-----.[----+<]-----.+++.+++[--+++<]-.) = bruh
```


~math ncr(20, 4)
```
ncr(20, 4) = 4845
```


~math plot_array([5, 9, 12, -1, 3])

![plot_array](https://cdn.discordapp.com/attachments/320915703102177293/815405611688525824/1614478245186992.png)

- This command is open to suggestions for more custom additions or integrations!


## Reminder
> ~reminder is a powerful command that can be used to schedule messages in an embed at a certain point in the future.

### Examples
![remind_3s](https://cdn.discordapp.com/attachments/320915703102177293/815406848076677140/unknown.png)

![remind_7h1m50s](https://cdn.discordapp.com/attachments/682553066209148942/815408094560518174/unknown.png)

![remind_5m55s](https://cdn.discordapp.com/attachments/320915703102177293/815408198344245258/unknown.png)

![remind_99999](https://cdn.discordapp.com/attachments/320915703102177293/815422717314334740/unknown.png)

![remind_3pm](https://cdn.discordapp.com/attachments/320915703102177293/815423163210268712/unknown.png)

![announce_miza](https://cdn.discordapp.com/attachments/320915703102177293/815423786928701440/unknown.png)

![remind_when](https://cdn.discordapp.com/attachments/320915703102177293/815424392338866186/unknown.png)

![remind_as](https://cdn.discordapp.com/attachments/320915703102177293/815424838482919454/unknown.png)

![remind_every](https://cdn.discordapp.com/attachments/320915703102177293/815425461202059274/unknown.png)
from prettytable import PrettyTable as ptable

def _m2f(mem,val):
    val1 = mem
    val2 = val
    curr = 0
    result = ""
    while val2:
        difference = int(val1,16)-4840864+curr/8
        flag = difference*8
        offset = max(0,int((-flag+999.9)/1000))
        flag += offset*1000
        output = ""
        for i in range(0,3):
            a = 10**i
            b = flag//a
            char = b%10
            char += 48
            output += chr(char)
        char = int(flag/1000)
        char += 48
        char -= offset
        if val2&1:
            operation = "+"
        else:
            operation = "-"
        try:
            output += chr(char)
            output = "<FL"+operation+output[::-1]
        except:
            output = "<FL"+operation+"(0x"+hex((char+256)&255).upper()[2:]+")"+output[::-1]
        result += output
        val2 >>= 1
        curr += 1
    return result
def pull_douclub(argv):
    c_id = _vars.auth["knack_id"]
    c_sec = _vars.auth["knack_secret"]
    baseurl = "https://api.knack.com/v1/pages/"#scene_xx/views/view_yy/records"
    items = argv.replace(" ","%20").lower()
    baseurl = "https://doukutsuclub.knack.com/database#search-database/?view_22_search="
    url = baseurl+items+"&view_22_page=1"
    print(url)
    resp = _vars.op.open(url)
    if resp.getcode() != 200:
        raise ConnectionError("Error "+str(resp.getcode()))
    s = resp.read().decode("utf-8")

    return s

class cs_mem2flag:
    is_command = True
    def __init__(self):
        self.name = ["cs_m2f"]
        self.minm = 0
        self.desc = "Returns a sequence of Cave Story TSC commands to set a certain memory address to a certain value."
        self.usag = '<0:address> <1:value[1]>'
    async def __call__(self,_vars,args,**void):
        if len(args) < 2:
            return _m2f(args[0],1)
        return _m2f(args[0],_vars.evalMath(" ".join(args[1:])))

class cs_npc:
    is_command = True
    def __init__(self):
        self.name = []
        self.minm = 0
        self.desc = "Searches the Cave Story NPC list for an NPC by name or ID."
        self.usag = '<0:query>'
    async def __call__(self,_vars,args,flags,**void):
        lim = ("c" in flags)*40+20
        argv = " ".join(args)
        data = _vars.ent.search(argv,lim)
        if len(data):
            head = _vars.ent.data[1]
            for i in range(len(head)):
                if head[i] == "":
                    head[i] = i*" "
            table = ptable(head)
            for line in data:
                table.add_row(line)
            output = str(table)
            if len(output)<10000 and len(output)>1900:
                response = ["Search results for **"+argv+"**:"]
                lines = output.split("\n")
                curr = "```\n"
                for line in lines:
                    if len(curr)+len(line) > 1900:
                        response.append(curr+"```")
                        curr = "```\n"
                    if len(line):
                        curr += line+"\n"
                response.append(curr+"```")
                return response
            else:
                return "Search results for **"+argv+"**:\n```\n"+output+"```"
        else:
            return "No results found for **"+argv+"**."

class cs_tsc:
    is_command = True
    def __init__(self):
        self.name = []
        self.minm = 0
        self.desc = "Searches the Cave Story OOB flags list for a memory variable."
        self.usag = '<0:query>'
    async def __call__(self,_vars,args,flags,**void):
        lim = ("c" not in flags)*40+20
        argv = " ".join(args)
        data = _vars.tsc.search(argv,lim)
        if len(data):
            head = _vars.tsc.data[0]
            for i in range(len(head)):
                if head[i] == "":
                    head[i] = i*" "
            table = ptable(head)
            for line in data:
                table.add_row(line)
            output = str(table)
            if len(output)<10000 and len(output)>1900:
                response = ["Search results for **"+argv+"**:"]
                lines = output.split("\n")
                curr = "```\n"
                for line in lines:
                    if len(curr)+len(line) > 1900:
                        response.append(curr+"```")
                        curr = "```\n"
                    if len(line):
                        curr += line+"\n"
                response.append(curr+"```")
                return response
            else:
                return "Search results for **"+argv+"**:\n```\n"+output+"```"
        else:
            return "No results found for **"+argv+"**."

class cs_mod:
    is_command = True
    def __init__(self):
        self.name = []
        self.minm = 0
        self.desc = "Searches the Doukutsu Club and Cave Story Tribute Site Forums for an item."
        self.usag = '<0:query>'
    async def __call__(self,**void):
        pass

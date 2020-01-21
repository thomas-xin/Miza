def mem2flag(mem,val):
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
            b = int((flag/a))
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

INF = 10**9

fed_2021_bracket = [
        (10, 19900),
        (12, 81050),
        (22, 172750),
        (24, 329850),
        (32, 418850),
        (35, 628300),
        (37, INF),
        ]

ca_2021_bracket = [
        (1, 17864),
        (2, 42350),
        (4, 66842),
        (6, 92788),
        (8, 117268),
        (9.3, 599016),
        (10.3, 718814),
        (11.3, 1198024),
        (12.3, INF),
        ]


def tax_from_bracket(brk, income):
    tax = 0
    for rate, ceil in brk:
        tax += rate/100.0 * min(ceil, income)
        income -= ceil
        if income <= 0:
            break
    return tax

def fed(income):
    return tax_from_bracket(fed_2021_bracket, income)

def ca(income):
    return tax_from_bracket(ca_2021_bracket, income)


def total_tax(income):
    fed_tax = fed(income)
    ca_tax = ca(income)
    all_tax = fed_tax + ca_tax
    print("income: %d" % (income))
    print("fed tax %d, %f%%" % (fed_tax, fed_tax * 100.0 / income))
    print("ca tax %d, %f%%" % (ca_tax, ca_tax * 100.0 / income))
    print("all tax %d, %f%%" % (all_tax, all_tax * 100.0 / income))
    print("")

total_tax(200000)
total_tax(300000)
total_tax(500000)
total_tax(750000)
total_tax(1000000)

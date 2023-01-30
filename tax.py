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

fed_2022_deduction = 25900
fed_2022_bracket = [
        (0, fed_2022_deduction),
        (10, 20550 + fed_2022_deduction),
        (12, 83550 + fed_2022_deduction),
        (22, 178150 + fed_2022_deduction),
        (24, 340100 + fed_2022_deduction),
        (32, 431900 + fed_2022_deduction),
        (35, 647850 + fed_2022_deduction),
        (37, INF),
        ]

fed_2022_social_bracket = [
        (6.2, 147000),
        (0, INF),
        ]
fed_social_bracket = fed_2022_social_bracket


fed_2022_medicare_bracket = [
        (1.45, INF),
        ]

fed_2022_amt_exemption = 118100
fed_2022_amt_bracket = [
        (0, fed_2022_amt_exemption),
        (26, 206100),
        (28, 1079800),
        (35, 1079800 + fed_2022_amt_exemption),
        (28, INF),
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

fed_bracket = fed_2022_bracket
fed_amt_bracket = fed_2022_amt_bracket


def tax_from_bracket(brk, income):
    #print('calculate tax')
    tax = 0
    prev_ceil = 0
    for rate, ceil in brk:
        amount = min(ceil - prev_ceil, income)
        #print('ceil and pre v ceil', ceil, prev_ceil)
        #print('amount', amount)
        tax += rate/100.0 * amount
        #print('now tax', tax)
        income -= amount
        prev_ceil = ceil
        #print('remain income', income)
        if income <= 0:
            break
    #print('final tax', tax)
    return tax

def pure_fed(income):
    income_t = tax_from_bracket(fed_bracket, income)
    return income_t

def fed(income):
    income_t = tax_from_bracket(fed_bracket, income)
    osadi = tax_from_bracket(fed_social_bracket, income)
    medicare = tax_from_bracket(fed_2022_medicare_bracket, income)
    return income_t + osadi + medicare

def fed_amt(income):
    return tax_from_bracket(fed_amt_bracket, income)

def ca(income):
    return tax_from_bracket(ca_2021_bracket, income)

def ca_tmt(income):
    return income * 0.0665


def total_tax(income):
    pure_fed_tax = pure_fed(income)
    fed_tax = fed(income)
    fed_amt_tax = fed_amt(income)
    ca_tax = ca(income)
    ca_amt = ca_tmt(income)
    all_tax = fed_tax + ca_tax
    print("income: %d" % (income))
    print("fed pure tax %d, %f%%" % (pure_fed_tax, pure_fed_tax * 100.0 / income))
    print("fed tax %d, %f%%" % (fed_tax, fed_tax * 100.0 / income))
    print("fed amt %d, %f%%" % (fed_amt_tax, fed_amt_tax * 100.0 / income))
    print("fed amt refund %d" % (pure_fed_tax - fed_amt_tax))
    print("ca tax %d, %f%%" % (ca_tax, ca_tax * 100.0 / income))
    print("ca tmt %d, %f%%" % (ca_amt, ca_amt * 100.0 / income))
    print("all tax %d, %f%%" % (all_tax, all_tax * 100.0 / income))
    print("")

print('fed', fed_bracket)
print('ca', ca_2021_bracket)

total_tax(200000)
total_tax(250000)
total_tax(300000)
total_tax(362198)
#total_tax(400000)
total_tax(500000)
total_tax(600000)
#total_tax(700000)
total_tax(750000)
#total_tax(800000)
total_tax(1000000)
#total_tax(1250000)
total_tax(1500000)
total_tax(2000000)
#total_tax(3000000)
#total_tax(4000000)
#total_tax(5000000)

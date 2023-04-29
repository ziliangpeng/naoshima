import matplotlib.pyplot as plt

import tax


FED_TAX = []
FED_AMT = []

FED_DIFF = []
FED_PCT = []
AGIs = []

for agi in range(1000, 2000000, 1000):
    fed_pure = tax.pure_fed(agi)
    fed_amt = tax.fed_amt(agi)

    FED_TAX.append(fed_pure)
    FED_AMT.append(fed_amt)
    FED_DIFF.append(fed_pure - fed_amt)
    FED_PCT.append((fed_pure - fed_amt)/agi)
    AGIs.append(agi)

# plt.plot(AGIs, FED_PCT)
plt.plot(AGIs, FED_DIFF)
plt.show()
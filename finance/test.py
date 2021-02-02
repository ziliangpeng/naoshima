import yfinance as yf
import pandas
from collections import defaultdict


def extract_close_price(data, i):
    return data['Close'][i]

def aggr_by_month(mark):
    return '-'.join(str(mark.date()).split('-')[0:2])

def aggr_by_year(mark):
    return '-'.join(str(mark.date()).split('-')[0:1])

def aggr_by_quarter(mark):
    y, m, = str(mark.date()).split('-')[0:2]
    q = (int(m) -1 ) // 3 + 1
    return str(y) + ' q' + str(q)

def avg(arr):
    return sum(arr) / len(arr)


def aggregate(raw_data, fn_extract=extract_close_price, fn_aggr_row=aggr_by_month, fn_aggr_num=avg):
    l = len(raw_data)
    d = defaultdict(list)
    for i in range(l):
        mark = raw_data.index[i]
        new_mark = fn_aggr_row(mark)
        d[new_mark].append(fn_extract(raw_data, i))

    d2 = {}
    for k, v in d.items():
        d2[k] = fn_aggr_num(v)

    return d2


names = ['ARKK', 'ARKQ', 'TSLA', 'AMZN', 'GOOG', 'FB', 'NFLX']

for name in names:
    print(name)
    stock = yf.Ticker(name)
    hist = stock.history(period="10y")


    d = aggregate(hist, fn_aggr_row=aggr_by_quarter)
    data = []
    up_back = 4
    for item in sorted(d.keys()):
        value = d[item]
        up = 0.0 if len(data) < up_back else value/data[-up_back]
        if 'q1' in item:
            print('%s  %.2f  %.2f' % (item, value, up))
        data.append(value)

    # close_prices = hist['Close']

    # length = len(hist.index)

    # # print(hist)
    # # print(dir(hist))
    # # print(type(hist))
    # # for h in hist['Open']:
    # #     print(h)
    # #     print(dir(h))
    # #     break

    # for i in range(0, length):
    #     date_str = hist.index[i].date()
    #     if not str(date_str).endswith('-02-01'):
    #         continue
    #     # hist.index[i]

    #     print(date_str, close_prices[i])
    #     # print(type(hist.index[i]))
    # # for c in close_prices:
    # #     print(c)
    # #     print(dir(c))
    # #     print(type(c))
    # #     break
    print()
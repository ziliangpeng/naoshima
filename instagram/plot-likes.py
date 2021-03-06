import sys
from os.path import commonprefix

import plotly.graph_objs as go
import plotly.offline as py

date_list = []


def make_data(dates, dict):
    return [d in dict and dict[d] or 0 for d in dates]


data_list = []
date_list = []
for fo_data_file in sys.argv[1:]:

    sub_date_list = []
    data_dict = {}
    with open(fo_data_file, 'r') as f:
        for line in f.readlines():
            date = ' '.join(line.split(' ')[1:6]).split(':')[0] + ':00'
            if len(date_list) > 0 and date not in date_list[0]:
                continue

            if len(line.split()) < 7:
                continue  # sum-of-likes is missing
            likes = int(line.split()[-1])
            sub_date_list.append(date)
            data_dict[date] = likes

    data_list.append(data_dict)
    date_list.append(sub_date_list)

prefix = commonprefix(sys.argv[1:])
suffix = commonprefix([s[::-1] for s in sys.argv[1:]])[::-1]
graphs = []
for i in range(len(data_list)):
    graphs.append(
        go.Scatter(
            x=date_list[i],
            y=make_data(date_list[i], data_list[i]),
            name=sys.argv[i + 1][len(prefix):-len(suffix)],
            mode='lines',
        )
    )

py.plot({
    "data": graphs,
}, filename='/data/reckless/likes-gen.html', auto_open=False)

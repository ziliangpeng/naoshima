import sys

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
            data_block = line.split('\"')[3]
            foer_cnt = int(data_block.split(' ')[0].replace(',', ''))
            sub_date_list.append(date)
            data_dict[date] = foer_cnt

    data_list.append(data_dict)
    date_list.append(sub_date_list)

graphs = []
for i in range(len(data_list)):
    graphs.append(
        go.Scatter(
            x=date_list[i],
            y=make_data(date_list[i], data_list[i]),
            name=sys.argv[i + 1],
            mode='lines',
        )
    )

py.plot({
    "data": graphs,
})

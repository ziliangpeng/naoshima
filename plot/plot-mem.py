import plotly.graph_objs as go
import plotly.offline as py

mem_filename = '/data/mem.txt'
swap_filename = '/data/swap.txt'

mem_date_list = []
total_mem_list = []
used_mem_list = []
with open(mem_filename, 'r') as f:
    for line in f.readlines():
        if '|' not in line:
            continue
        date_line = line.split('|')[1].strip()
        date = ':'.join(date_line.split(':')[:2]) + ':00'
        mem_date_list.append(date)

        mem_block = line.split('|')[0]
        mem_data = mem_block.split()
        total_mem = mem_data[1]
        used_mem = mem_data[2]
        total_mem_list.append(total_mem)
        used_mem_list.append(used_mem)


swap_date_list = []
used_swap_list = []
with open(swap_filename, 'r') as f:
    for line in f.readlines():
        print('line', line)
        if '|' not in line:
            continue
        date_line = line.split('|')[1].strip()
        date = ':'.join(date_line.split(':')[:2]) + ':00'
        print('date', date)
        swap_date_list.append(date)

        swap_block = line.split('|')[0]
        swap_data = swap_block.split()
        total_swap = swap_data[1]
        used_swap = swap_data[2]
        used_swap_list.append(used_swap)


total_mem_graph = \
    go.Scatter(
        x=mem_date_list,
        y=total_mem_list,
        name='total mem',
        mode='lines',
    )
used_mem_graph = \
    go.Scatter(
        x=mem_date_list,
        y=used_mem_list,
        name='used mem',
        mode='lines',
    )
used_swap_graph = \
    go.Scatter(
        x=swap_date_list,
        y=used_swap_list,
        name='used swap',
        mode='lines',
    )

graphs = [total_mem_graph, used_mem_graph, used_swap_graph]

py.plot({
    "data": graphs,
}, filename='/data/reckless/mem.html')

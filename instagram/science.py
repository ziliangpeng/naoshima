import sys

import plotly.graph_objs as go
import plotly.offline as py

import auth
import utils


def gen():
    bot = auth.auth()
    with open('science.txt', 'w') as f:
        # __________________v___
        for fid, fname in utils.get_all_followers_gen(bot, 2288001113):
            followed_by, follows = utils.get_follow_counts(fname)
            f.write('%s %d %d\n' % (fname, followed_by, follows))
            f.flush()


def plot():
    xs, ys = [], []
    with open('science.txt', 'r') as f:
        for line in f.readlines():
            fname, followed_by, follows = line.split()
            xs.append(min(7500, int(followed_by)))
            ys.append(min(7500, int(follows)))

    # Create a trace
    trace = go.Scatter(
        x=xs,
        y=ys,
        mode='markers',
        marker=dict(
            size=2,
        )
    )

    data = [trace]

    # Plot and embed in ipython notebook!
    py.plot(data, filename='science_followers')


if __name__ == '__main__':
    if sys.argv[1] == 'gen':
        gen()
    elif sys.argv[1] == 'plot':
        plot()

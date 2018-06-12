import web
import os
import datadog


urls = (
    '/', 'index',
    '/hyperclips', 'hyperclips',
    # '/sex', 'sex',
    # '/baby', 'baby'
)

DD_HOST_KEY = 'DD_HOST'

if DD_HOST_KEY in os.environ:
    dd_host = os.getenv(DD_HOST_KEY)
    sd = datadog.DogStatsd(host=dd_host)
else:
    sd = datadog.statsd


class index:
    def GET(self):
        sd.increment('naoshima.reckless.index.count')
        raise web.seeother('/static/main.html')


class hyperclips:
    def GET(self):
        raise web.seeother('/static/hyperclips/hyper.html')


if __name__ == "__main__":
    print('Starting server...')
    app = web.application(urls, globals())
    app.internalerror = web.debugerror
    app.run()

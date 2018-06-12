import web
import datadog


urls = (
    '/', 'index',
    '/hyperclips', 'hyperclips',
    '/sex', 'sex',
    '/baby', 'baby'
)


class index:
    def GET(self):
        datadog.DogStatsd.increment('naoshima.reckless.index.count')
        raise web.seeother('/static/main.html')


class hyperclips:
    def GET(self):
        raise web.seeother('/static/hyperclips/hyper.html')


if __name__ == "__main__":
    print('Starting server...')
    app = web.application(urls, globals())
    app.internalerror = web.debugerror
    app.run()

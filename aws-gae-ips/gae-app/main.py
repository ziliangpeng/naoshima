import os
import datetime
import logging
from google.appengine.ext.webapp import template
from google.appengine.ext import webapp
from google.appengine.ext.webapp.util import run_wsgi_app
from google.appengine.ext import db
from google.appengine.api import users

import urllib2

ROUTE_DOMAIN = 'http://news.ycombinator.com'

class StartPage(webapp.RequestHandler):

  def get(self):
    route_url = ROUTE_DOMAIN + self.request.path + '?' + self.request.query_string
    req = urllib2.Request(url=route_url, headers={ 'User-Agent' : 'Mozilla/5.0' })
    raw_html = urllib2.urlopen(req).read()
    self.response.out.write(raw_html)


application = webapp.WSGIApplication([('/.*', StartPage)],
                                     debug=True)


def main():
  run_wsgi_app(application)


if __name__ == "__main__":
  main()

# Project N. Crawling HN.

## Mark 1

Since Google App Engine uses a pool of IPs for urlopen, it avoids being banned from HN.

This test is to set up a server in AWS, and set up massive urlfetch in GAE, to find out IP use pattern of GAE.

**Result**:

see ips.txt for IP log

run ip-analysis.py to see stats.

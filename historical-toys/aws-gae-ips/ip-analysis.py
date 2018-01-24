from collections import defaultdict

m = defaultdict(int)

lines = open('ips.txt').readlines()
for line in lines:
	ip = line[:line.find('g')-1]

	m[ip] += 1
	print ip

print 'sums'
for k, v in sorted(m.items()):
	print k, '%4d' % v#, float(v)/len(lines)


def solve(path):
	s = set()
	i, j = 0, 0
	for c in path:
		s.add((i, j, c))
		if c == 'S':
			i += 1
		elif c == 'E':
			j += 1
		else:
			exit(1)

	ans = ""
	i, j = 0, 0
	for _ in range(0, len(path), 2):
		# SE
		if (i, j, 'S') not in s and (i + 1, j, 'E') not in s:
			ans += "SE"
			i += 1 
			j += 1
			continue
		# ES
		elif (i, j, 'E') not in s and (i, j + 1, 'S') not in s:
			ans += "ES"
			i += 1
			j += 1
			continue
		else:
			print("error no SE or ES")
			exit(1)

	return ans


T = input()

for _ in range(T):
	N = input()
	path = raw_input()
	print("Case #%d: %s" % (_ + 1, solve(path)))

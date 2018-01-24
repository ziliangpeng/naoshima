
def read_array(convertor=None):
    ret = raw_input().split()
    if convertor: ret = map(convertor, ret)
    return ret


def main():
	n = input()
	a = read_array(int)
	m = input()
	for i in range(m):
		x, y = read_array(int)
		x -= 1
		y -= 1
		if x > 0:
			a[x-1] += y
		if x < n - 1:
			a[x+1] += a[x] - y - 1
		a[x] = 0
	for i in range(n):
		print a[i]



main()


def read_array(convertor=None):
  ret = raw_input().split()
  if convertor: ret = map(convertor, ret)
  return ret


def main():
	n, ns, m ,ms = input(), read_array(int), input(), read_array(int)
	ans = 0
	cnt = 0
	for i in ns:
		for j in ms:
			if j % i == 0:
				tmp = j / i
				if tmp > ans:
					ans = tmp
					cnt = 0
				if tmp == ans:
					cnt += 1

	print cnt



main()

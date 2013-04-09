
def read_array(convertor=None):
    ret = raw_input().split()
    if convertor: ret = map(convertor, ret)
    return ret


def main():
	n = input()
	books1 = []
	books2 = []
	for i in range(n):
		t, w = read_array(int)
		if t == 1:
			books1.append(w)
		else:
			books2.append(w)

	books1 = list(reversed(sorted(books1)))
	books2 = list(reversed(sorted(books2)))

	ans = 1000000000
	for num_1 in range(len(books1)+1):
		for num_2 in range(len(books2)+1):
			total_thick = num_1 + num_2 * 2
			sum_w = sum(books1[num_1:]) + sum(books2[num_2:])
			if total_thick >= sum_w and total_thick < ans:
				ans = total_thick

	print ans

main()

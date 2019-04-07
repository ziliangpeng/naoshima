
T = int(raw_input())

for _ in range(T):
	N = raw_input()
	A = list(N)
	for i in range(len(A)):
		if A[i] == '4':
			A[i] = '3'
	A = ''.join(A)

	n = int(N)
	a = int(A)
	b = n - a

	print("Case #%d: %d %d" % (_ + 1, a, b))




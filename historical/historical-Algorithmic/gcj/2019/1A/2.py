from collections import defaultdict
import sys

def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)
	pass

T, N, M = map(int, input().split())
eprint(T, N, M)
# T test case
# N number of guess, 365 for small, 7 for large
# M max number of gopher

PRIMES = [2,3,5,7,11,13,17]

for t in range(1, 1 + T):
	remainder_dict = {}

	for p in PRIMES:
		print(' '.join([str(p)] * 18))
		# eprint('guessed ' + ' '.join([str(p)] * 18))
		sys.stdout.flush()

		judge_return = input()
		# eprint('get ' + judge_return)
		if judge_return == '-1':
			exit(1)
		else:
			remainders = sum(map(int, judge_return.split(' ')))
			remainder_dict[p] = remainders % p
			# eprint('sum remainder ' + str(remainders % p))

	found = False
	ans = -1
	count = defaultdict(int)
	for p in PRIMES:
		remainder = remainder_dict[p]
		for i in range(M):
			val = i * p + remainder
			count[val] += 1
			if count[val] == len(PRIMES):
				ans = val;
				found = True
				break
			if val > M:
				break

	# eprint('count ' + str(count[101]))
	# eprint(str(count))
	# for ret in range(1, M+1):
	# 	# eprint('ret ' + str(ret))
	# 	if count[ret] == len(PRIMES):
	# 		print(ret)
	# 		judge_score = input()
	# 		if judge_score == '-1':
	# 			eprint('wrong :-(')
	# 		eprint('ans ' + str(ret))
	# 		found = True
	# 		sys.stdout.flush()
	# 		sys.stderr.flush()
	# 		break
	if not found:
		print(42)
	elif found:
		print(ans)
		eprint(count[888888])
		judge_score = input()
		eprint('ans ' + str(ans))



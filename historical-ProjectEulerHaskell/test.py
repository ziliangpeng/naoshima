from os import listdir
from re import match
import subprocess
from datetime import datetime
from termcolor import colored

""" Auto test program to verify program for each locked problem can run, and can generate correct answer."""

# TODO: warning when a solution runs slow

solutions = map(lambda x:x[:-3], filter(lambda x:match('^\d+\.hs$', x), listdir('.')))

answers = map(lambda x:x[:-4], filter(lambda x:match('^\d+\.ans$', x), listdir('./ans')))

success_cnt = 0
fail_cnt = 0

for prob_id in answers:
    print 'problem %s' % prob_id

    if prob_id not in solutions:
        print 'solution not exist' % prob_id
        continue

    with open('ans/' + prob_id + '.ans') as f:
        answer = f.read().strip()

    time_before = datetime.now()
    result = subprocess.check_output(["./run.sh", prob_id])
    time_after = datetime.now()
    result = result.split()[-1].strip() # only interested in the last line which is the answer
    print 'running time %s' % str(time_after - time_before)
    print 'expected %16s, get %16s,' % (answer, result),
    if result == answer:
        print colored('success', 'green') + '!'
        success_cnt += 1
    else:
        print colored('fail', 'red') + '!'
        fail_cnt += 1
    print ''

    solutions.remove(prob_id)

print '%d tested,' % (success_cnt + fail_cnt),

if success_cnt + fail_cnt == 0:
    print 'nothing to test.'
elif fail_cnt == 0:
    print colored('all passed', 'green') + '!'
else:
    print success_cnt, colored(success_cnt, 'green'),
    print fail_cnt, colored(fail_cnt, 'red')

if solutions:
    print 'untested solutions (because of lack of answer):', solutions

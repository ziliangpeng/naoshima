import gflags
import glog
import sys

FLAGS = gflags.FLAGS
gflags.DEFINE_integer('init_age', 36, '')
gflags.DEFINE_float('init_w', 1.0, 'in million')
gflags.DEFINE_float('start_rate', 1.15, '')
gflags.DEFINE_float('end_rate', 1.05, '')
gflags.DEFINE_float('init_spend', 0.1, '')
gflags.DEFINE_float('income', 0.2, '')
gflags.DEFINE_float('inflate', 1.04, '')
gflags.DEFINE_integer('retire_age', 50, '')

arglist = sys.argv
try:
    remaining_args = gflags.FLAGS(argv=arglist)
except gflags.Error as e:
    print('%s\nUsage:\n%s' % (e, gflags.FLAGS))
    sys.exit(1)
glog.init()

rate = FLAGS.start_rate
spend = FLAGS.init_spend
w = FLAGS.init_w


print('%8s   %8s   %8s   %8s   %8s   %8s' % ('YEAR', 'AGE', 'SPEND', 'WEALTH', 'RATE', 'DIFF'))
print('\n')
for age in range(FLAGS.init_age, 100):
  prev_w = w
  w *= rate
  w -= spend
  if age <= FLAGS.retire_age:
    w += FLAGS.income 
  rate -= (FLAGS.start_rate - FLAGS.end_rate) / 60
  #print(1985 + age, age, spend, w, rate)
  print('%8d   %8d   %8f   %8f   %8f   %8f' % (1985+age, age, spend, w, rate, w - prev_w))
  spend *= FLAGS.inflate


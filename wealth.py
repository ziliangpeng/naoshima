init_age = 36 # 40
init_w = 1.5 # 10 # in million

start_rate = 1.20 # capital gain
end_rate = 1.05
rate = start_rate
spend = 0.13
inflate = 1.04

retire_age = 55
income = 0.1

w = init_w
for age in range(init_age, 100):
  w *= rate
  w -= spend
  if age <= retire_age:
    w += income 
  rate -= (start_rate - end_rate) / 60
  print(1985 + age, age, spend, w, rate)
  spend *= inflate


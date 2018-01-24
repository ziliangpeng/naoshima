puts " >>> if statement"
x = 3
if x.even?
    puts x.to_s + " is even"
else
    puts x.to_s + " is odd"
end

puts "#{x} is multiple of 5" if x % 5 == 0 # if modifier
puts "#{x} is not multiple of 5" if x % 5 != 0
puts "#{x} is something" unless x == 0 # unless modifier

case x
when 3, 5, 7
    puts "#{x} is holy number"
when -1..1
    puts "#{x} very small"
when 0...10
    puts "#{x} is single digit"
else
    puts "nothing special"
end

puts ' >>> while loop'
y = 12345678987654321
i = 1
while i < y do
    i *= 123
    puts "i = #{i}"
end

i /= 2 while i > 100
puts "i is #{i}"

begin
    i *= 999
    puts "i eq #{i}"
end while i < y

puts 'we can also use `until` modifier to do opposite of `while`'

for i in 1..3
    puts "i = #{i} in for loop"
end

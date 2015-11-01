puts "something at line 1"

print "something at line 3\n"

BEGIN { puts "Initializing at line 5" }

END { puts "Ending at line 7" }

puts "something at line 9"


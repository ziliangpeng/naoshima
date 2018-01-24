cmd = [
    '123',
    '1_23_4',
    '0x88',
    '0b101',
    '?a',
    '?\n',
    '3.2e4',
    ]

cmd.each do |command|
    puts "==> Command: #{command}"
    puts "Result: #{eval command}"
    puts ""
end

# And now for something slightly different

puts ' >>> literal string. no escape.'
puts '123\t\n'
puts ' >>> except these'
puts '123\'4\'. we need \\'

puts ' >>> double quote is used for escape'
puts "1\t2"
puts ' >>> double quote also used for expr'
puts "We can have result for expression: 60*60*24 = #{60 * 60 * 24}"


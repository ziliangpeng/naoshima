cmd = [
    '1.methods',
    '"Abc".methods',
    '1.even?',
    '2.next',
    '3.class',
    '"xoxo".class'
    ]

cmd.each do |command|
    puts "==> Command: #{command}"
    puts "Result: #{eval command}"
    puts ""
end

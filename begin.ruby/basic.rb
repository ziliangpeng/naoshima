cmd = [
    '1.methods',
    '1.even?',
    '2.next']

cmd.each do |command|
    puts "==> Command: #{command}"
    puts "Result: #{eval command}"
    puts ""
end

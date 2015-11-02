=begin
block is a code chunk you put in {}, with a name immediately before it.
block is invoked by a call to a method with the same name.
with yield in that method, code can be executed coordinately.
=end
def test
    yield 5
    puts "inside test method 1"
    yield 9
    puts "inside test method 2"
    yield '1st val of 2 vals', '2nd val of 2 vals'
    puts "inside test method 3"
    yield
end

test  {|x| puts "in block with yield value \"#{x}\""}

=begin
BEGIN and END block is special blocks that will execute before and after program execution.
=end

puts "===================== SEPARATE ======================"
# You can have method that takes parameter
def nonsense name
    yield 1
    puts "#{name} is a non sense person"
    yield 2
    puts "but #{name} is very nice"
    yield 3
    puts "he makes people happy"
end

nonsense("Steve B.") {|x| puts "chapter #{x}"}


BEGIN {
    puts "!!!!! THIS IS THE BEGINNING OF PROGRAM EXECUTION"
}
END {
    puts "!!!!! THIS IS THE END OF PROGRAM EXECUTION"
}

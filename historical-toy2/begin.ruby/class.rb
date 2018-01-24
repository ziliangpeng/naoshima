class Technology

    IS_ART = false # Constant

    $planet = 'Earth' # Global variable

    @@name = 'Future' # class variable

    @age = 'right now' # instance variable

    def initialize(age, year=2015, location)
        @age = age
        @year = year
        @location = location
    end

    def display
        puts $planet
        puts @@name
        puts @age
        puts @year
        puts @location
    end
end


tech = Technology. new('Info Age', location='China') # All whitespaces are ignored

tech.display

puts 'IS_ART: ' + Technology::IS_ART.to_s # must use :: for correct namespace
Technology::IS_ART = true ## can modify, but gives warning
puts 'IS_ART: ' + Technology::IS_ART.to_s

print 'Printing the function object itself: '
puts tech.method(:display) # get the functioin object itself


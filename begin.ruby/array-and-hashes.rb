
nums = [1, '2', "3", 4.0, ?5]
nums.each do |x|
    puts x
end
puts 'Arrays support wild range of functional methods, like drop, take, count, any, map, each, index, reverse, rotate, ...'
puts ''

puts 'can use range for enumerating numbers. `..` is inclusive, `...` is exclusive'
(99..103).each do |x|
    puts 'range number ' + x.to_s
end

mappings = {'Google' => 'Search', 'Facebook' => 'Social', 'Airbnb' => 'Hospitality'}
mappings.each do |k, v|
    print 'Company: ' + k + ', business: ' + v + "\n"
end

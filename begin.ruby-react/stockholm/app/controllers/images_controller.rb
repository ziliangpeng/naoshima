class Image
  def initialize(url)
    @url = url
  end
end
class ImagesController < ApplicationController
  def index
    @images = [
      Image.new('https://c2.staticflickr.com/6/5491/14097130247_ec80ae9dd0_k.jpg'),
      Image.new('http://c2.staticflickr.com/8/7044/7148620445_7148c368bf_h.jpg'),
    ]
  end
end

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/trusty64"
  config.vm.hostname = "kecume.box"
  config.vm.network :private_network, ip: "192.168.5.7"

  config.vm.provision "chef_solo" do |chef|
    chef.channel = "stable"
    chef.version = '12.10.24'
    chef.add_recipe "kecume-base"
  end
end

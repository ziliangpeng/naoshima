#!/bin/zsh

mkdir initconverge

for hostname in $(curl http://optica.d.musta.ch | jq -r ".nodes[] | .hostname" | grep -E 'i-[0-9a-f]{8}') 
do
    echo $hostname
    if [[ -a initconverge/$hostname.init.local ]]; then
        echo 'skip'
        continue
    else
        ssh -A -o StrictHostKeyChecking=no -o NumberOfPasswordPrompts=0 $hostname.inst.aws.airbnb.com cat /var/log/init > "initconverge/$hostname.init.local"
    fi
done

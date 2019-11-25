import os
import subprocess
import json
import shutil


def get_creation_date(path):
    # ffprobe -v quiet {filename} -print_format json -show_entries format_tags=creation_time 
    cmd = ['ffprobe', '-v', 'quiet', path, '-print_format', 'json', '-show_entries', 'format_tags=creation_time']
    out = subprocess.Popen(cmd, 
           stdout=subprocess.PIPE, 
           stderr=subprocess.STDOUT)
    stdout,_ = out.communicate()
    js = json.loads(stdout)
    creation_time = js['format']['tags']['creation_time']
    return creation_time


def convert(path, creation_time):
    # ffmpeg -i inputfile.mp4 -metadata date="$(stat --printf='%y' inputfile.mp4 | cut -d ' ' -f1)" -codec copy outputfile.mp4
    # cmd = ['ffmpeg', '-i', path, '-metadata', 'date='+create_time, '-codec', 'copy', 'outputfile.mp4']
    # out = os.Popen(cmd, 
    #        stdout=subprocess.PIPE, 
    #        stderr=subprocess.STDOUT)
    # stdout,_ = out.communicate()
    out_file = '%s.%s.converted.mp4' % (path, creation_time)
    if os.path.isfile(out_file):
        return
    out_file_tmp = out_file + '.tmp.mp4'
    os.system('ffmpeg -i %s -map_metadata g %s' % (path, out_file_tmp))
    shutil.move(out_file_tmp, out_file)

for path in os.listdir('.'):
    if 'converted.mp4' not in path.lower():
        creation_time = get_creation_date(path)
        print('creation time of %s is %s' % (path, creation_time))
        # convert(path, creation_time)
        convert(path, creation_time)

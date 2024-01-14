import os
import click

from mutagen.mp4 import MP4

def find_files_of_old_and_new(old_dir, new_dir):
    old_dir_files, new_dir_files = {}, {}

    def find_all(d):
        second_level_dirs = ["Apple Music", "Music"]
        file_list = {}
        for second_level_dir in second_level_dirs:
            for prefix, _, fs in os.walk(os.path.join(d, second_level_dir)):
                for f in fs:
                    full_file_path = os.path.join(prefix, f)[len(d)+1:]
                    file_list[full_file_path] = 1
        return file_list

    old_dir_files = find_all(old_dir)
    new_dir_files = find_all(new_dir)

    return old_dir_files, new_dir_files


@click.command()
@click.option('--new_dir')
@click.option('--old_dir')
def main(new_dir, old_dir):
    old_dir_files, new_dir_files = find_files_of_old_and_new(old_dir, new_dir)
    print(f"New dir files: {len(new_dir_files)}")
    print(f"Old dir files: {len(old_dir_files)}")

    overlaps = []
    for f in new_dir_files:
        if f.endswith('m4a') and f in old_dir_files:
            overlaps.append(f)
    assert len(overlaps) != 0

    file_path = os.path.join(new_dir, overlaps[0])
    audio = MP4(file_path)
    # Print metadata
    print(os.path.getsize(file_path))
    for key, value in audio.tags.items():
        print(f'{key}: {value}')

    file_path = os.path.join(old_dir, overlaps[0])
    audio = MP4(file_path)
    # Print metadata
    print(os.path.getsize(file_path))
    for key, value in audio.tags.items():
        print(f'{key}: {value}')


    

if __name__ == '__main__':
    main()
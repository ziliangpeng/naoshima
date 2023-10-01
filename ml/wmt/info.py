# import wget

# # Download the WMT dataset
# url = 'http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz'
# filename = wget.download(url)

# # Open the downloaded file and read its contents
# with open(filename, 'r') as f:
#     contents = f.read()

# # Print the contents of the file
# print(contents)

import tensorflow_datasets as tfds

# Load the WMT dataset
dataset, info = tfds.load('wmt13_translate/cs-en', with_info=True)

# Print the dataset info
print(info)
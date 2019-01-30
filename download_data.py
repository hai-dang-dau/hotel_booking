from __future__ import print_function

import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

train_url = 'https://www.dropbox.com/s/zwwwgpktwhkn8g7/train.csv?dl=1'
test_url = 'https://www.dropbox.com/s/qepk94f5z09fb3k/test.csv?dl=1'


def main(output_dir='data'):
    filenames = ['train.csv', 'test.csv']
    urls = [train_url, test_url]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # notfound = []
    for url, filename in zip(urls, filenames):
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            print("{} already exists".format(output_file))
            continue

        print("Downloading from {} ...".format(url))
        urlretrieve(url, filename=output_file)
        print("=> File saved as {}".format(output_file))


if __name__ == '__main__':
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()

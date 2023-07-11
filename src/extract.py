import os
from subprocess import run
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('--srcpath', type=str, 
                       help='the path where you download the packed raw data in')
argparser.add_argument('--tarpath', type=str, 
                       help='the target path where you want to extract the packed raw data to')

if __name__ == '__main__':
     # parse arguments from the console
    args = argparser.parse_args()
    
    files = os.listdir(args.srcpath)
    
    files.sort()
    for file in files:
         if ".zip" not in file:
              continue
         zip_file = os.path.join(args.srcpath, file)
         
         proc = run(['unzip', zip_file, '-d', args.tarpath])
         
    print("SUCCESSFULLY extracted all raw data to %s" %(os.path.join(args.tarpath, 'data')))
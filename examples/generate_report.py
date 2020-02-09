import argparse
from utils import load_chain
from system_utils import generate_report

parser = argparse.ArgumentParser(description = 'Specify the name of the obj to be loaded')
parser.add_argument('-f', '--file_name', metavar='', type = str,
  help = 'The name of the file of the object to be loaded.')
parser.add_argument('-t', '--report_title', metavar='', type = str,
  help = 'Title of the report to be genrated.')
parser.add_argument('-s', '--auprc', metavar='', type = str,
  help = 'Area under the precision recall curve.')

args = parser.parse_args()

def main():
  network = load_chain(args.file_name)

  generate_report(network.network_args, args.report_title, args.auprc)
  
if __name__ == '__main__':
  main()
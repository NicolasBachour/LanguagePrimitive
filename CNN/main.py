
import sys
import parser

def main(argv):
    if (len(argv) < 2):
        print("Please specify the folder containing the dataset.")
        return
    data = parser.load_dataset(argv[1])

    #for line in content:
        #print("\t" + line)
    #content = [x.strip() for x in content]
    #print("\t" + filename + " in " + folder)

    return

if __name__ == "__main__":
    main(sys.argv)

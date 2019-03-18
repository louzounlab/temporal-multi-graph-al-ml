"""
If the motif (or any other feature) calculations turn out to be incorrect,
this code removes the incorrect pickle files.
"""
import os


def remove_file(file_name):
    base_dir = os.path.join(os.getcwd(), 'graph_measures')
    for f1 in os.listdir(base_dir):
        print(f1)
        for f2 in os.listdir(os.path.join(base_dir, f1)):
            print(f2)
            for f3 in os.listdir(os.path.join(base_dir, f1, f2)):
                print(f3)
                os.remove(os.path.join(base_dir, f1, f2, f3, file_name + '.pkl'))


if __name__ == "__main__":
    remove_file('motif3')
    remove_file('motif4')


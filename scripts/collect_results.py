import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)
import re
import pandas as pd
from tabulate import tabulate


metrics_rec = {}
metrics_gen = {}
all_results = []
# setting device on GPU if available, else CPU
dir_name= os.path.join(main_dir_path,'Checkpoints/')

metrics_rec = {}
metrics_gen = {}
subdirs = [x[0] for x in os.walk(dir_name)]
all_results = []
for subdir in subdirs: 
    result_dict = {}
    try:
        # Try to open the file in read mode
        file= "reconstruction_metrics.txt"
        # Create a list to store the results
        result_dict['Model']=subdir
        with open(os.path.join(subdir, file), "r") as file:
            # Read the contents of the file
            file_rec = file.read()
            # Define a regex pattern to match text and numbers
            pattern = r'([A-Za-z\s,]+): (\d+\.\d+) %'

            # Find all matches in the input string
            matches = re.findall(pattern, file_rec)

            # Iterate over the matches and extract text and numbers
            for match in matches:
                text = match[0].strip()
                number = float(match[1])
                result_dict[text] = number

        # Try to open the file in read mode
        file= "generated_polymers.txt"
        with open(os.path.join(subdir, file), "r") as file:
            # Read the contents of the file
            file_gen = file.read()
                        # Define a regex pattern to match text and numbers
            pattern = r'([A-Za-z\s,]+): (\d+\.\d+) %'

            # Find all matches in the input string
            matches = re.findall(pattern, file_gen)


            # Iterate over the matches and extract text and numbers
            for match in matches:
                text = match[0].strip()
                number = float(match[1])
                result_dict[text] = number
        all_results.append(result_dict)
    except FileNotFoundError:
        print(f"The file {file} does not exist in subdir {subdir}")
    


df = pd.DataFrame(all_results)
df.to_csv(dir_name+'/all_results.csv', sep=';')    
print(tabulate(df, headers='keys', tablefmt='psql'))



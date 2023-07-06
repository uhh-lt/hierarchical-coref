# code to split conll files in 4 parts


import os

def count_sentences(file_path):
    sentence_count = 0

    with open(file_path, 'r') as file:
        for line in file:
            line = line.lstrip()  # Remove leading/trailing whitespace
            if not line or line[0] == '#':
                continue
            columns = line.split()
            if int(columns[2]) == 0:
                sentence_count += 1
                
    return sentence_count

def split_conll_file(file_path, output_directory):
    sentence_count = count_sentences(file_path)
    k1 = 0.25  # Split ratio 
    k2 = 0.25
    k3 = 0.25
    # k4 = 0.25
    
    split1_sentence_count = int(sentence_count * 0.25)
    split2_sentence_count = int(sentence_count * 0.25)
    split3_sentence_count = int(sentence_count * 0.25)

    # Extracting the document name and directory from the file path
    document_dir, document_name = os.path.split(file_path)
    document_name = os.path.splitext(document_name)[0]

    # Creating directory paths for the split files
    split_dir = os.path.join(output_directory, "")
    os.makedirs(split_dir, exist_ok=True)

    # Creating file names for the split files
    file1_path = os.path.join(split_dir, '{}_1.gold_conll'.format(document_name))
    file2_path = os.path.join(split_dir, '{}_2.gold_conll'.format(document_name))
    file3_path = os.path.join(split_dir, '{}_3.gold_conll'.format(document_name))
    file4_path = os.path.join(split_dir, '{}_4.gold_conll'.format(document_name))

    with open(file_path, 'r') as input_file, \
         open(file1_path, 'w') as output_file1, \
         open(file2_path, 'w') as output_file2, \
         open(file3_path, 'w') as output_file3, \
         open(file4_path, 'w') as output_file4:

        current_sentence_count = 0
        current_output_file = output_file1
        current_output_file.write('#begin document ({}); part 0\n'.format(file1_path[len(split_dir):-11]))
        
        for line in input_file:
            line = line.lstrip()
            if not line: 
                current_output_file.write('\n')
                continue
            if line[0] == '#':
                continue
                
            columns = line.split()
            
            if int(columns[2]) == 0:
                current_sentence_count += 1

                if current_sentence_count == split1_sentence_count:
                    current_output_file.write('#end document\n')
                    current_output_file = output_file2
                    current_output_file.write('#begin document ({}); part 0\n'.format(current_output_file.name[len(split_dir):-11]))

                elif current_sentence_count == split1_sentence_count + split2_sentence_count:
                    current_output_file.write('#end document\n')
                    current_output_file = output_file3
                    current_output_file.write('#begin document ({}); part 0\n'.format(current_output_file.name[len(split_dir):-11]))

                elif current_sentence_count == split1_sentence_count + split2_sentence_count + split3_sentence_count:
                    current_output_file.write('#end document\n')
                    current_output_file = output_file4
                    current_output_file.write('#begin document ({}); part 0\n'.format(current_output_file.name[len(split_dir):-11]))

            line = line.replace(f"{document_name}", f"{current_output_file.name[len(split_dir):-11]}")
            current_output_file.write(line)
        current_output_file.write('#end document' + '\n')
        print("Split complete. Splitted files: '{}' ".format(current_output_file.name))


# input_file = 'litbank/test/74_the_adventures_of_tom_sawyer_brat.gold_conll'
# output_path = 'litbank_splitted_4/'

# split_conll_file(input_file, output_path)

input_directories = ['litbank/development', 'litbank/train', 'litbank/test']
output_directories = ['litbank_splitted_4/development', 'litbank_splitted_4/train', 'litbank_splitted_4/test']

for i, input_directory in enumerate(input_directories):
    output_directory = output_directories[i]

    # Iterate over files in the input directory
    for filename in os.listdir(input_directory):
        file_path = os.path.join(input_directory, filename)
        if os.path.isfile(file_path):
            split_conll_file(file_path, output_directory)

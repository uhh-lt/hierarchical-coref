import os


def count_sentences(file_path):
    sentence_count = 0

    with open(file_path, 'r') as file:
        for line in file:
            line = line.lstrip()  # Remove leading/trailing whitespace
            if not line or line[0] == '#':
                continue
            columns = line.split()
            # if(len(columns) < 3): continue
            #print(len(columns[2]))
            if int(columns[2]) == 0:
                
                sentence_count += 1
                
    return sentence_count

def split_conll_file(file_path, output_directory):
    sentence_count = count_sentences(file_path)
    k = 0.6
    half_sentence_count = int(sentence_count * k)

    # Extracting the document name and directory from the file path
    document_dir, document_name = os.path.split(file_path)
    document_name = os.path.splitext(document_name)[0]

    # Creating directory paths for the split files
    split_dir = os.path.join(output_directory, "")
    os.makedirs(split_dir, exist_ok=True)

    # Creating file names for the split files
    
    file1_path = os.path.join(split_dir, '{}_1.gold_conll'.format(document_name))
    file2_path = os.path.join(split_dir, '{}_2.gold_conll'.format(document_name))


    # # Extracting the document name from the file path
    # document_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # # Creating file names for the split files
    # file1_path = '{}_1.gold_conll'.format(document_name)
    # file2_path = '{}_2.gold_conll'.format(document_name)

    with open(file_path, 'r') as input_file, \
         open(file1_path, 'w') as output_file1, \
         open(file2_path, 'w') as output_file2:

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
            
            # columns[0] = current_output_file.name[:-11]
            # line = ' '.join(columns) + '\n'
            
            if int(columns[2]) == 0:
                current_sentence_count += 1

                # Switch output file when half the sentences are reached
                if current_sentence_count == half_sentence_count:
                    current_output_file.write('#end document\n')
                    current_output_file = output_file2
                    current_output_file.write('#begin document ({}); part 0\n'.format(file2_path[:-11]))


            line = line.replace(f"{document_name}", f"{current_output_file.name[len(split_dir):-11]}")
            current_output_file.write(line)
        current_output_file.write('#end document' + '\n')
    print("Split complete. Split files: '{}' and '{}'".format(file1_path, file2_path))



# file_path = 'litbank/dev/36_the_war_of_the_worlds_brat.gold_conll'
# split_conll_file(file_path)

input_directories = ['litbank/dev', 'litbank/train', 'litbank/test']
output_directories = ['litbank_splitted/dev', 'litbank_splitted/train', 'litbank_splitted/test']

for i, input_directory in enumerate(input_directories):
    output_directory = output_directories[i]

    # Iterate over files in the input directory
    for filename in os.listdir(input_directory):
        file_path = os.path.join(input_directory, filename)
        if os.path.isfile(file_path):
            split_conll_file(file_path, output_directory)


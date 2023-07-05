# file to remove heads from a jsonlines and write to another .jsonlines file


import jsonlines

input_file = 'litbank_splitted/jsonlines/english_test.jsonlines'
output_file = 'litbank_splitted_no_head/english_test.jsonlines'

with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
    for line in reader:

        line['head'] = [None] * len(line['head'])
        writer.write(line)

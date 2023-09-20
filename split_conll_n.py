"""
Split script for droc dataset
"""
import argparse
import os
from pathlib import Path

def count_sentences(file_path):
    sentence_count = 0

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line or line[0] == '#':
                continue
            columns = line.split()
            try:
                new_word_count, *_ = columns
                new_word_count = int(new_word_count)
            except ValueError:
                _, _, new_word_count, *_ = columns
                new_word_count = int(new_word_count)
            if new_word_count == 1:
                sentence_count += 1
    return sentence_count


def split_conll_file(file_path, output_directory, num_splits):
    split_ratio = 1.0 / num_splits
    sentence_count = count_sentences(file_path)
    min_per_split = int(sentence_count * split_ratio)

    # Extracting the document name and directory from the file path
    _document_dir, document_name = os.path.split(file_path)
    document_name = os.path.splitext(document_name)[0]

    # Creating directory paths for the split files
    split_dir = os.path.join(output_directory, "")
    os.makedirs(split_dir, exist_ok=True)

    out_paths = [os.path.join(split_dir, '{}_{}.gold_conll'.format(document_name, x)) for x in range(1, num_splits + 1)]

    with open(file_path, 'r') as input_file:
        current_sentence_index = 0
        output_files = [open(p, 'w') for p in out_paths]
        current_file_index = int(current_sentence_index / min_per_split)
        current_output_file = output_files[current_file_index]
        name = '#begin document ({}); part 0\n'.format(current_output_file.name[len(split_dir):-11])
        output_files[current_file_index].write(name)
        for line in input_file:
            if line.startswith("#"):
                continue
            line = line.lstrip()
            current_file_index = int(current_sentence_index / min_per_split)
            if current_file_index == len(output_files): # Whatever is left goes into the last file
                current_file_index -= 1
            current_output_file = output_files[current_file_index]
            if line == "":
                current_sentence_index += 1
                if current_sentence_index % min_per_split == 0:
                    current_output_file.write('#end document\n')
                    if current_file_index < len(output_files) - 1:
                        current_output_file = output_files[current_file_index + 1]
                        name = '#begin document ({}); part 0\n'.format(current_output_file.name[len(split_dir):-11])
                        output_files[current_file_index + 1].write(name)
                else:
                    current_output_file.write('\n')
            else:
                current_output_file.write(line)

        output_files[-1].write('#end document\n')
        print("Split complete. Split files:", *out_paths)

def main(input_dir, output_dir, num_splits):
    splits = ["development", "train", "test"]
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_directories = [input_dir / split for split in splits]
    output_directories = [output_dir / split for split in splits]

    for i, input_directory in enumerate(input_directories):
        output_directory = output_directories[i]

        # Iterate over files in the input directory
        for filename in os.listdir(input_directory):
            file_path = os.path.join(input_directory, filename)
            if os.path.isfile(file_path):
                split_conll_file(file_path, output_directory, num_splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("num_splits")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, int(args.num_splits))
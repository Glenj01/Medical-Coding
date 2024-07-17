import csv

def remove_empty_lines(input_file, output_file):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            line_count = 0
            for row in reader:
                # Skip empty lines (lines with no content)
                if not any(row):
                    continue
                writer.writerow(row)
                
                # Print progress every 10000 lines
                line_count += 1
                if line_count % 10000 == 0:
                    print("Processed", line_count, "lines")

if __name__ == "__main__":
    input_file = 'notes_labeled.csv'
    output_file = 'cleaned_notes_labeled.csv'
    remove_empty_lines(input_file, output_file)
    print("Empty lines removed and saved to:", output_file)
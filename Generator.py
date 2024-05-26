import pandas as pd
import random
import xlsxwriter


def generate_random_numbers(count, start, end):
    """Generate a list of random numbers."""
    return [random.randint(start, end) for _ in range(count)]


def save_to_excel(data, filename):
    """Save the data to an Excel file."""
    df = pd.DataFrame(data, columns=['Random Numbers'])
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()
    print(f"{filename} has been saved.")


if __name__ == "__main__":
    # Generate 1000 random numbers between 1 and 100,000
    random_numbers = generate_random_numbers(10000, 1, 100000)

    # Save the random numbers to an Excel file
    save_to_excel(random_numbers, 'random_numbers10000.xlsx')

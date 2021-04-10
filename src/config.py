from pathlib import Path

file_dir = Path(__file__).resolve().parent

class CONFIG:
    data = file_dir.parent / 'data'
    report = file_dir.parent / 'report'
    notebook = file_dir.parent / 'notebook'
    src = file_dir.parent / 'src'

if __name__ == '__main__':
    print(file_dir)


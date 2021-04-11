from pathlib import Path

file_dir = Path(__file__).resolve().parent

class CONFIG:
    data = file_dir.parent.parent / 'data'
    report = file_dir.parent.parent / 'report'
    notebook = file_dir.parent.parent / 'notebook'
    src = file_dir.parent.parent / 'src'

if __name__ == '__main__':
    print(file_dir)


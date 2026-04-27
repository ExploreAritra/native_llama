import sys
import re

def fix_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match data arrays and their following length constants
    # Group 1: array name and size
    # Group 2: array data
    # Group 3: following length constant
    
    # We use a non-greedy match for data, but it needs to be robust.
    # The data ends with "};" followed by "const uint64_t <name>_len = <value>;"
    
    pattern = re.compile(r'(const unsigned char \w+\[\d+\] = \{)(.*?)(\s*const uint64_t \w+_len = \d+;)', re.DOTALL)
    
    def replace_func(match):
        header = match.group(1)
        data = match.group(2).strip()
        footer = match.group(3)
        
        # If data doesn't end with };, add it.
        # But wait, the pattern might consume multiple arrays if we are not careful.
        # Actually, the data usually ends with something like "0xXX,0xXX," or "0xXX"
        
        if not data.endswith('};'):
             return f"{header}\n{data}\n}};\n\n{footer}"
        return match.group(0)

    # Alternative: just find every "const uint64_t ..._len" and ensure the preceding non-whitespace is "};"
    lines = content.splitlines()
    new_lines = []
    for i, line in enumerate(lines):
        if line.startswith('const uint64_t ') and '_len =' in line:
            # Look back
            j = len(new_lines) - 1
            while j >= 0 and not new_lines[j].strip():
                j -= 1
            if j >= 0 and new_lines[j].strip() != '};' and not new_lines[j].startswith('#include') and not new_lines[j].startswith('const uint64_t'):
                new_lines.insert(j + 1, '};')
                new_lines.insert(j + 2, '')
        new_lines.append(line)
        
    # Also handle the very end of file arrays if any
    # (though usually the arrays are followed by _len constants or tables)
    
    # Let's also check for the tables at the end
    tables = ["const void * add_data", "const uint64_t add_len", "const void * add_rms_data", "const uint64_t add_rms_len"]
    final_lines = []
    for i, line in enumerate(new_lines):
        for table in tables:
            if line.startswith(table):
                j = len(final_lines) - 1
                while j >= 0 and not final_lines[j].strip():
                    j -= 1
                if j >= 0 and final_lines[j].strip() != '};' and not final_lines[j].startswith('const uint64_t'):
                     final_lines.insert(j + 1, '};')
                     final_lines.insert(j + 2, '')
                break
        final_lines.append(line)

    with open(file_path, 'w') as f:
        f.write('\n'.join(final_lines) + '\n')

if __name__ == "__main__":
    fix_file(sys.argv[1])

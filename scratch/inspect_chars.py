with open('D:/github/github/ofn-ddos-detector/DOKUMENTACJA_NAUKOWA.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()
line = lines[75].strip()
print("Line:", repr(line))
for i, c in enumerate(line):
    print(f"{i}: {repr(c)} (ord: {ord(c)})")

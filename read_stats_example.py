stats_file='preprocessed_hist_record.txt'

with open('preprocessed_hist_record.txt') as f:
    raw = f.read()
    formatted = eval(raw)

# now formatted is a list
type(formatted)
len(formatted)


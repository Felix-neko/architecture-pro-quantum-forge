from more_itertools import chunked

in_list = list(range(10))
print(list(chunked(in_list, 4)))

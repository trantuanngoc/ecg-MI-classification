

def iterate_fold(arr):
    for i in range(len(arr)):
        current = [arr[i]]
        remaining = arr[:i] + arr[i+1:]
        yield current, remaining




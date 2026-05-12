def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
    
def sum(n):
    if n == 0:
        return 0
    else:
        return n + sum(n - 1)

def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)

def reverse_string(s):
    if s == "":
        return s
    else:
        return reverse_string(s[: 1, -1]) + s[0]

def count_down(n):
    if n <= 0:
        return n
    else:
        print(n)
        return count_down(n - 1)

if __name__ == "__main__":
    x = factorial(5)
    y = sum(10)
    z = fib(15)
    s = reverse_string("Hello, World!")
    print(count_down(5))
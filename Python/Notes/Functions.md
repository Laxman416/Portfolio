# Functions

## Docstrings

"""
Description: What fn does

Args:

Returns:


"""

`/__.doc__`
`inspect.getdoc()`
**Dont Repeat and Do One Thing**

pass by assignment

integers are immutable - cant be changed by fn

## Context Managers
Set up context
Run code 
Remove context

```python
with open() -> #context manager
```

**Writing Context Managers:**
- Class-based
- Function-based

```python
@contextlib.contextmanager
def fn():
    # code need
    yield
    # teardown code

```

**Advanced Topics:**

*Nested contexts:*

```python
@contextlib.contextmanager
def copy(src, dst):

    with open(src) as f:
        with open(dst, 'w') as f_dst:
            for line in f:
                f_dst.write(line)
    
    try:
        yield
    finally:
        # tear down code
```

## Decorators

Fn another object
Global and local scope
non local is in parent fn used in child fn
closures: fn contains tuple of memory
- non local variables in parent fn to returned when calling child fn

Wrapper around fn

```python
@double_args
def multiply(a,b)
    return a * b
# Decorator will double args -> 4* fn

def double_args(func):
    def wrapper(a,b):

        return func(a * 2, b * 2)

    return wrapper
 
multiply = double_args(multiply) #don't need use decorator syntax
multiply(1,5) -> # 20
```

```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        t_start = time.time()
        result = func(*args, **kwargs)

        t_total = time.time() - t_start
        print(t_total)
        return reult
    return wrapper
```

```python
def memoize(func):
    # Store results in dict that maps arguments to results, doesnt create new as always in closure
    cache = {}

    @wraps(func) -> # modify wrapper metadeta to look like calling fn meta data
    def wrapper(*args, **kwargs):
        # key for kwargs
        kwargs_key = tuple(sorted(kwargs.items()))
        if (args, kwargs_key) not in cache:
            # store result
            cache[(args, kwargs_key)] = func(*args, **kwargs)
        return cache[(args, kwargs_key)]
    return wrapper
```
When to use:
- common code to all fn

Metadata:
use `from functools import wraps`

easy access to undecorated fn using `__.wrapped__`

**Decorators that take arguments**

```python
def run_n_times(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator
    
@run_n_times(3)
def print_sum(a,b)
    print(a+b)

```

**Timeout()**

```python
import signal

def timeout_in_5s(n_seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.alarm(5)
        try:
            # raise Timeout Error if too long
            return func(*args, **kwargs)
        finally:
            # cancel alarm
            signal.alarm(0)
        return wrapper
    return decorator


```
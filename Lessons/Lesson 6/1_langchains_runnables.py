from functools import partial

from langchain.schema.runnable import RunnableLambda

################################################################################
## Very simple "take input and return it"
identity = RunnableLambda(lambda x: x)


################################################################################
## Given an arbitrary function, you can make a runnable with it
def print_and_return(x, preface=""):
    print(f"{preface}{x}")
    return x


rprint = RunnableLambda(print_and_return)

################################################################################
## Chaining two runnables
chain1 = identity | rprint
output = chain1.invoke("Hello World!")
print()


################################################################################
## You can also pre-fill some of values using functools.partial
rprint1 = RunnableLambda(partial(print_and_return, preface="1: "))


################################################################################
## And you can use the same idea to make your own custom Runnable generator
def RPrint(preface=""):
    return partial(print_and_return, preface=preface)


################################################################################
## Chaining that one in as well
output = (
    chain1  ## Prints "Welcome Home!" & passes "Welcome Home!" onward
    | rprint1  ## Prints "1: Welcome Home!" & passes "Welcome Home!" onward
    | RPrint("2: ")  ## Prints "2: Welcome Home!" & passes "Welcome Home!" onward
).invoke("Welcome Home!")

# Final Output Is Preserved As "Welcome Home!"
print("\nOutput:", output)

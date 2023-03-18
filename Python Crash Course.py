#!/usr/bin/env python
# coding: utf-8

# # Python Crash Course Exercises 
# 
# This is an optional exercise to test your understanding of Python Basics. If you find this extremely challenging, then you probably are not ready for the rest of this course yet and don't have enough programming experience to continue. I would suggest you take another course more geared towards complete beginners, such as [Complete Python Bootcamp](https://www.udemy.com/complete-python-bootcamp)

# ## Exercises
# 
# Answer the questions or complete the tasks outlined in bold below, use the specific method described if applicable.

# ** What is 7 to the power of 4?**

# In[ ]:


7**4


# ** Split this string:**
# 
#     s = "Hi there Sam!"
#     
# **into a list. **

# In[ ]:


text = "Hi there Sam!"
x = text.split()


# In[ ]:


print(x)


# ** Given the variables:**
# 
#     planet = "Earth"
#     diameter = 12742
# 
# ** Use .format() to print the following string: **
# 
#     The diameter of Earth is 12742 kilometers.

# In[ ]:


planet = "Earth"
diameter = 12742


# In[ ]:


print("The diameter of {p} is {d} kilometers".format(p=planet,d=diameter))


# ** Given this nested list, use indexing to grab the word "hello" **

# In[ ]:


lst = [1,2,[3,4],[5,[100,200,['hello']],23,11],1,7]


# In[ ]:


lst[3][1][2][0]


# ** Given this nested dictionary grab the word "hello". Be prepared, this will be annoying/tricky **

# In[ ]:


d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}


# In[ ]:


d['k1'][3]['tricky'][3]['target'][3]


# ** What is the main difference between a tuple and a list? **

# In[ ]:


Tupple is immutable


# ** Create a function that grabs the email website domain from a string in the form: **
# 
#     user@domain.com
#     
# **So for example, passing "user@domain.com" would return: domain.com**

# In[ ]:


def getDomain(s):
    return s.split('@')[1]


# In[ ]:


getDomain("user@domain.com")


# ** Create a basic function that returns True if the word 'dog' is contained in the input string. Don't worry about edge cases like a punctuation being attached to the word dog, but do account for capitalization. **

# In[ ]:


def findDog(s):
    return "dog" in s.lower()


# In[ ]:


findDog("Is there any Dog there ?")


# ** Create a function that counts the number of times the word "dog" occurs in a string. Again ignore edge cases. **

# In[ ]:


def countDog(st):
    count = 0
    for word in st.lower().split():
        if word == 'dog':
            count += 1
    return count


# In[ ]:


countDog('This dog runs faster than the other dog dude!')


# ** Use lambda expressions and the filter() function to filter out words from a list that don't start with the letter 's'. For example:**
# 
#     seq = ['soup','dog','salad','cat','great']
# 
# **should be filtered down to:**
# 
#     ['soup','salad']

# In[ ]:


seq = ['soup','dog','salad','cat','great']


# In[ ]:


seq = ['soup','dog','salad','cat','great']
list(filter(lambda s: s[0]=='s',seq))


# ### Final Problem
# **You are driving a little too fast, and a police officer stops you. Write a function
#   to return one of 3 possible results: "No ticket", "Small ticket", or "Big Ticket". 
#   If your speed is 60 or less, the result is "No Ticket". If speed is between 61 
#   and 80 inclusive, the result is "Small Ticket". If speed is 81 or more, the result is "Big    Ticket". Unless it is your birthday (encoded as a boolean value in the parameters of the function) -- on your birthday, your speed can be 5 higher in all 
#   cases. **

# In[ ]:


def ticket(speed, birthday):
    if birthday is True:
        speed -= 5
    if speed > 80:
        return "Big Ticket"
    elif speed >60:
        return "Small Ticket"
    else :
        return "No Ticket"


# In[ ]:


ticket(66, True)
ticket(81, False)


# In[ ]:





# # Great job!

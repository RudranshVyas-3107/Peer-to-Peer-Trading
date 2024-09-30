#!/usr/bin/env python
# coding: utf-8

# In[1]:


class AverageUsagePJ:
    def __init__(self):
        self.usage_values = []  # To store the latest usage values

    def update_usage(self, new_usage_value):
        self.usage_values.append(new_usage_value)
        if len(self.usage_values) > 5:
            self.usage_values.pop(0)  # Remove the oldest value to keep the list size to 5

    def calculate_average_usage(self):
        if not self.usage_values:
            return 0  # In case there are no usage values
        return sum(self.usage_values) / len(self.usage_values)


# In[ ]:





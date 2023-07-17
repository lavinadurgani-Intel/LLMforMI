from io import StringIO
import pandas as pd 

import openai
openai.api_key = "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"

def generate_response(input_text):
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=input_text,
    temperature=0,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
text = "A table summarizing the partnerships in the confidential computing domain in last 10 years\n\n| Company1 | Company2 | Year | Partnership Type |"
generate_response(text)
print(type(response.choices[0].text))
print(response.choices[0].text)
resp_table_text = response.choices[0].text

#resp_table_text = "| -------- | -------- | ---- | --------------- | | Microsoft | Intel | 2019 | Confidential Computing | | Google | Intel | 2018 | Confidential Computing | | IBM | Intel | 2017 | Confidential Computing | | Microsoft | AMD | 2016 | Confidential Computing | | Google | ARM | 2015 | Confidential Computing | | IBM | ARM | 2014 | Confidential Computing | | Microsoft | Qualcomm | 2013 | Confidential Computing | | Google | Qualcomm | 2012 | Confidential Computing | | IBM | Qualcomm | 2011 | Confidential Computing | | Microsoft | NVIDIA | 2010 | Confidential Computing |"

# Reading String in form of csv file
resp_table_text_list = resp_table_text.split("||")
print("String into List:\n",resp_table_text_list)
df = [n.split('|') for n in resp_table_text_list]
df = pd.DataFrame(df[1:], columns=df[0])
print("List into Dataframe:\n",df)

## Dataframe Approach 
#resp_table_text_string = StringIO(resp_table_text)
#df=pd.read_csv(resp_table_text_string, sep="| |")
#print("String into Dataframe:\n",df)